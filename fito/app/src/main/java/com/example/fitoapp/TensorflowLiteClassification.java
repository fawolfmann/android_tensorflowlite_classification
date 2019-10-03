package com.example.fitoapp;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import android.graphics.Paint;
import android.graphics.Canvas;

public class TensorflowLiteClassification {
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    /** MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;

    /** Dimensions of inputs. */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;

    /** Number of results to show in the UI. */
    private static final int MAX_RESULTS = 3;

    protected Interpreter tflite;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    protected ByteBuffer imgData = null;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
     * of the super class, because we need a primitive array here.
     */
    private float[][] labelProbArray = null;
    /** Labels corresponding to the output of the vision model. */
    private List<String> labels;

    public TensorflowLiteClassification(Activity activity) throws IOException {

        tfliteModel = loadModelFile(activity);
        tflite = new Interpreter(tfliteModel);
        labels = loadLabelList(activity);
        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE
                                * getImageSizeX()
                                * getImageSizeY()
                                * DIM_PIXEL_SIZE
                                * getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new float[1][getNumLabels()];
    }

    protected void runInference() {
        tflite.run(imgData, labelProbArray);
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labels = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        return labels;
    }

    protected void addPixelValue(int pixelValue) {
        imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < getImageSizeX(); ++i) {
            for (int j = 0; j < getImageSizeY(); ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
    }

    public static Bitmap prepareImage(Bitmap bitmap) {
        Paint paint = new Paint();

        Bitmap bmpBitmap = Bitmap.createScaledBitmap(
                bitmap,
                getImageSizeX(),
                getImageSizeY(),
                false);
        Canvas canvas = new Canvas(bmpBitmap);
        canvas.drawBitmap(bmpBitmap, 0, 0, paint);
        return bmpBitmap;
    }

    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");
        croppedBitmap = Bitmap.createScaledBitmap(bitmap, getImageSizeX(), getImageSizeY(), false);
        Trace.beginSection("preprocessBitmap");
        convertBitmapToByteBuffer(croppedBitmap);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("runInference");
        long startTime = SystemClock.uptimeMillis();
        runInference();
        long endTime = SystemClock.uptimeMillis();
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < labels.size(); ++i) {
            pq.add(
                    new Recognition(
                            "" + i,
                            labels.size() > i ? labels.get(i) : "unknown",
                            getNormalizedProbability(i),
                            null));
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection();
        return recognitions;
    }

    public static int getImageSizeX() {
        return 160;
    }

    public static int getImageSizeY() {
        return 160;
    }

    /** Preallocated buffers for storing image data in. */
    private final int[] intValues = new int[getImageSizeX() * getImageSizeY()];

    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        return "flower_MobileNetV2_01.tflite";
    }

    protected String getLabelPath() {
        return "flower_labels.txt";
    }

    protected float getNormalizedProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }

    protected int getNumLabels() {
        return labels.size();
    }

    protected float getProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }


}
