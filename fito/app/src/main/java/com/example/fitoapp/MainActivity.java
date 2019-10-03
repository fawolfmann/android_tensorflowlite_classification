package com.example.fitoapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public TextView TV1;
    public ImageView iV1;
    private int PICK_IMAGE_REQUEST = 1;
    private TensorflowLiteClassification tfclassification;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.content_main);
        TV1 = (TextView) findViewById(R.id.textView1);
        iV1 = (ImageView) findViewById(R.id.imageView1);
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
        if (resultCode == RESULT_OK) {
            try
            {
                final Uri imageUri = data.getData();
                final InputStream imageStream = getContentResolver().openInputStream(imageUri);

                final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                final List<TensorflowLiteClassification.Recognition> results = tfclassification.recognizeImage(selectedImage);
                TV1.setText(results.get(0).toString());
                setPicture(selectedImage);

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
}
