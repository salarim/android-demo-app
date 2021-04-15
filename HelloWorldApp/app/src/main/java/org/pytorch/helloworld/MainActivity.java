package org.pytorch.helloworld;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.PyTorchAndroid;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

import androidx.appcompat.app.AppCompatActivity;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

public class MainActivity extends AppCompatActivity {
  private final String FILE_NAME = "keyFeatures.txt";
  private Module module = null;
  List<float[]> keyFrameFeatures = new ArrayList<float[]>();
  String[] keyFramePaths = null;
  String[] queryPaths = null;
  int queryIndex = 0;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    module = PyTorchAndroid.loadModuleFromAsset(getAssets(), "mobile_model.pt");

    loadKeyFrameFeatures();
    Log.d("output", "keyFrameFeatures length: " + keyFrameFeatures.size());

    Bitmap queryBitmap = loadQuery();

    showClosestKeyFrame(queryBitmap);
  }

  public float[] runModule(Bitmap bitmap){
    long startTime = System.nanoTime();

    bitmap = Bitmap.createScaledBitmap(bitmap, 640, 480, false);
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
    float[] feature = outputTensor.getDataAsFloatArray();

    long endTime = System.nanoTime();
    long duration = (endTime - startTime) / 1000000;
    Log.d("output", "runModule duration (ms): " + duration);
    return feature;
  }

  private void loadKeyFrameFeatures() {
    try {
      keyFramePaths = getAssets().list("key_frames");
      String jsonString = loadFile();
      if (jsonString.equals("")) {
        for (String keyFramePath : keyFramePaths) {
          Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("key_frames/" + keyFramePath));
          float[] feature = runModule(bitmap);
          keyFrameFeatures.add(feature);
        }

        Gson gson = new Gson();
        jsonString = gson.toJson(keyFrameFeatures);
        saveToFile(jsonString);
      } else {
        Gson gson = new Gson();
        Type listOfMyFloats = new TypeToken<ArrayList<float[]>>() {}.getType();
        keyFrameFeatures = gson.fromJson(jsonString, listOfMyFloats);
      }
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading keyframes", e);
      finish();
    }

  }

  private Bitmap loadQuery() {
    Bitmap bitmap = null;
    try {
      if (queryPaths == null) queryPaths = getAssets().list("queries");

      bitmap= BitmapFactory.decodeStream(getAssets().open("queries/" + queryPaths[queryIndex]));
      ((ImageView) findViewById(R.id.image)).setImageBitmap(bitmap);
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading query", e);
      finish();
    }

    return bitmap;
  }

  private void showClosestKeyFrame(Bitmap queryBitmap) {
    float[] queryFeature = runModule(queryBitmap);

    int closestInd = getClosestKeyFrameIndex(queryFeature);

    Bitmap closestBitmap = null;
    try {
      closestBitmap = BitmapFactory.decodeStream(getAssets().open("key_frames/" + keyFramePaths[closestInd]));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }
    ((ImageView) findViewById(R.id.image2)).setImageBitmap(closestBitmap);
  }

  public void nextButtonFunc(View v) {
    queryIndex = (queryIndex + 1) % queryPaths.length;

    Bitmap queryBitmap = loadQuery();
    showClosestKeyFrame(queryBitmap);
  }

  private int getClosestKeyFrameIndex(float[] queryFeature) {
    float min_dis = Float.MAX_VALUE;
    int min_ind = -1;

    for (int i=0; i < keyFrameFeatures.size(); i++) {
      float[] keyFrameFeature = keyFrameFeatures.get(i);
      float dis = 0.0f;
      for (int j=0; j < queryFeature.length; j++) {
        dis += Math.pow(queryFeature[j] - keyFrameFeature[j], 2);
      }

      if (dis < min_dis) {
        min_dis = dis;
        min_ind = i;
      }
    }

    return min_ind;
  }

  private void saveToFile(String s) {
    FileOutputStream fos = null;
    try {
      fos = openFileOutput(FILE_NAME, MODE_PRIVATE);
      fos.write(s.getBytes());
      Toast.makeText(this, "Saved to " + getFilesDir() + "/" + FILE_NAME,
              Toast.LENGTH_LONG).show();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (fos != null) {
        try {
          fos.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }
  }

  public String loadFile() {
    FileInputStream fis = null;
    try {
      fis = openFileInput(FILE_NAME);
      InputStreamReader isr = new InputStreamReader(fis);
      BufferedReader br = new BufferedReader(isr);
      StringBuilder sb = new StringBuilder();
      String text;
      while ((text = br.readLine()) != null) {
        sb.append(text).append("\n");
      }
      Toast.makeText(this, "Loaded from " + getFilesDir() + "/" + FILE_NAME,
              Toast.LENGTH_LONG).show();
      return sb.toString();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (fis != null) {
        try {
          fis.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }
    return "";
  }
}
