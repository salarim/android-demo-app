package org.pytorch.helloworld;

import android.content.ClipData;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

  Button select;
  int PICK_IMAGE_MULTIPLE = 1;
  ArrayList<String> mArrayUri = new ArrayList<String>();;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    select = findViewById(R.id.select);

    // click here to select image
    select.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {

        // initialising intent
        Intent intent = new Intent();

        // setting type to select to be image
        intent.setType("image/*");

        // allowing multiple image to be selected
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_MULTIPLE);
      }
    });
  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    // When an Image is picked
    if (requestCode == PICK_IMAGE_MULTIPLE && resultCode == RESULT_OK && null != data) {
      // Get the Image from data
      if (data.getClipData() != null) {
        ClipData mClipData = data.getClipData();
        int cout = data.getClipData().getItemCount();
        for (int i = 0; i < cout; i++) {
          // adding imageuri in array
          Uri imageurl = data.getClipData().getItemAt(i).getUri();
          mArrayUri.add(imageurl.toString());
        }
      } else {
        Uri imageurl = data.getData();
        mArrayUri.add(imageurl.toString());
      }
    } else {
      // show this if no image is selected
      Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
    }

    Bundle b = new Bundle();
    String[] arrayUri = new String[mArrayUri.size()];
    arrayUri = mArrayUri.toArray(arrayUri);
    b.putStringArray("KeyFrameUris", arrayUri);
    Intent intent = new Intent(this, ImageClassificationActivity.class);
    intent.putExtras(b);
    startActivity(intent);
  }
}
