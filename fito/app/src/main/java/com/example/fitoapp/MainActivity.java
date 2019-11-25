package com.example.fitoapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import java.io.File;
import android.content.Context;
import android.graphics.Canvas;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
// import com.yalantis.ucrop.UCrop;
import com.theartofdev.edmodo.cropper.CropImage;



public class MainActivity extends AppCompatActivity {

    public TextView TV1;
    public ImageView iV1;
    public ConstraintLayout constraintLayout;
    private int PICK_IMAGE_REQUEST = 100;
    private TensorflowLiteClassification tfclassification;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.content_main);
        TV1 = findViewById(R.id.textView1);
        iV1 = findViewById(R.id.imageView1);
        constraintLayout =  findViewById(R.id.layout1);
        try {
            tfclassification = new TensorflowLiteClassification(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void selectPhoto(View v)
    {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int reqCode, int resultCode, Intent data) {
        super.onActivityResult(reqCode, resultCode, data);

        if (resultCode == RESULT_OK && reqCode == PICK_IMAGE_REQUEST) {
            final Uri sourceUri = data.getData();
            File file = getImageFile(this); // 2
            Uri destinationUri = Uri.fromFile(file);
            openCropActivity(sourceUri, destinationUri);
        }

        // if (reqCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
        if (reqCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE && resultCode == RESULT_OK) {
            try
            {
                if (data != null) {
                    //Uri imageUri = UCrop.getOutput(data);
                    Uri imageUri = CropImage.getActivityResult(data).getUri();
                    final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                    final Bitmap tempBitMap = BitmapFactory.decodeStream(imageStream);
                    final Bitmap selectedImage = pad(tempBitMap);
                    final List<TensorflowLiteClassification.Recognition> results = tfclassification.recognizeImage(selectedImage);
                    TV1.setText(results.get(0).toString());
                    setPicture(selectedImage);
                }

        } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
    //set Picture in the ImageView (iV1)
    public void setPicture(Bitmap bp)
    {
        Bitmap scaledBp =  Bitmap.createScaledBitmap(bp, iV1.getWidth(), iV1.getHeight(), false);
        iV1.setImageBitmap(scaledBp);
    }

    private void openCropActivity(Uri sourceUri, Uri destinationUri) {
        CropImage.activity(sourceUri)
                .start(this);
        /*UCrop.Options options = new UCrop.Options();
        // options.setCircleDimmedLayer(true);
        UCrop.of(sourceUri, destinationUri)
                .withMaxResultSize(1920, 1920)
                .withAspectRatio(0, 0)
                .start(this);*/
    }

    private File getImageFile(Context context) {
        String imageFileName = "JPEG_" + System.currentTimeMillis() + "_";
        File storageDir = context.getCacheDir();
        File file = null;
        try {
            file = File.createTempFile(
                    imageFileName, ".jpg", storageDir
            );
        } catch (IOException e) {
            e.printStackTrace();
        }
        return file;
    }

    public Bitmap pad(Bitmap Src) {
        int new_height, new_width;
        float top, left;

        if (Src.getWidth() > Src.getHeight() ){
            new_height = Src.getWidth();
            new_width = Src.getWidth();
            top = (Src.getWidth() - Src.getHeight()) >> 1;
            left = 0.f ;
        }else {
            new_height = Src.getHeight();
            new_width = Src.getHeight();
            top = 0.f ;
            left = (Src.getHeight() - Src.getWidth()) / 2;
        }
        Bitmap outputImage = Bitmap.createBitmap(new_width ,new_height, Bitmap.Config.ARGB_8888);
        Canvas can = new Canvas(outputImage);
        can.drawARGB(255,0,0,0); //This represents Black color
        can.drawBitmap(Src, left, top, null);
        return outputImage;
    }
}
