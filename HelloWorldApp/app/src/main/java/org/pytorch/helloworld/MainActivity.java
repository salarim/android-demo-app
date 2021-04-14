package org.pytorch.helloworld;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.PyTorchAndroid;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
  private Module module = null;
  List<float[]> keyFrameFeatures = new ArrayList<float[]>();
  String[] keyFramePaths = null;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    module = PyTorchAndroid.loadModuleFromAsset(getAssets(), "mobile_model.pt");

    Bitmap queryBitmap = null;
    try {
      queryBitmap = BitmapFactory.decodeStream(getAssets().open("queries/IMG_0731.JPG"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    ((ImageView) findViewById(R.id.image)).setImageBitmap(queryBitmap);

    loadKeyFrameFeatures();
    Log.d("output", "keyFrameFeatures length: " + keyFrameFeatures.size());

    float[] queryFeature = runModule(queryBitmap);

    int closestInd = getClosestKeyFrameIndex(queryFeature);
    Log.d("output", "closest file is: " + keyFramePaths[closestInd]);

    Bitmap closestBitmap = null;
    try {
      closestBitmap = BitmapFactory.decodeStream(getAssets().open("key_frames/" + keyFramePaths[closestInd]));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }
    ((ImageView) findViewById(R.id.image2)).setImageBitmap(closestBitmap);

  }

  public float[] runModule(Bitmap bitmap){
    bitmap = Bitmap.createScaledBitmap(bitmap, 640, 480, false);
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    return outputTensor.getDataAsFloatArray();
  }

  private void loadKeyFrameFeatures() {
    try {
      keyFramePaths = getAssets().list("key_frames");
      for (String keyFramePath: keyFramePaths) {
        Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("key_frames/" + keyFramePath));
        float[] scores = runModule(bitmap);
        keyFrameFeatures.add(scores);
      }
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading keyframes", e);
      finish();
    }
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
}
