package org.pytorch.helloworld;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.ImageView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;
import androidx.exifinterface.media.ExifInterface;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

    private static final int INPUT_TENSOR_WIDTH = 640;
    private static final int INPUT_TENSOR_HEIGHT = 480;
    private static final String FILE_NAME = "keyFeatures.txt";

    static class AnalysisResult {

        private final int closestInd;

        public AnalysisResult(int closestInd) {
            this.closestInd = closestInd;
        }
    }

    private Module mModule;
    List<float[]> keyFrameFeatures = new ArrayList<float[]>();
    String[] keyFramePaths = null;

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_image_classification;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
                .inflate()
                .findViewById(R.id.image_classification_texture_view);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Bundle b=this.getIntent().getExtras();
        keyFramePaths = b.getStringArray("KeyFrameUris");
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        Bitmap closestBitmap = null;
        try {
            Uri keyFrameUri = Uri.parse(keyFramePaths[result.closestInd]);
            closestBitmap = rotateBitmap(keyFrameUri);
        } catch (IOException e) {
            Log.e("PyTorchDemo", "Error reading assets", e);
            finish();
        }
        ((ImageView) findViewById(R.id.closestImage)).setImageBitmap(closestBitmap);
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mModule == null) {
            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "mobile_model.pt");
            Log.d("PyTorchDemo", "Module loaded!");

            loadKeyFrameFeatures();
            Log.d("PyTorchDemo", "Key frame features loaded! len: " + keyFrameFeatures.size());
        }

        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, true);

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
        float[] feature = outputTensor.getDataAsFloatArray();

        int closestInd = getClosestKeyFrameIndex(feature);

        return new AnalysisResult(closestInd);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mModule != null) {
            mModule.destroy();
        }

        String filePath = getFilesDir() + "/" + FILE_NAME;
        File fdelete = new File(filePath);
        if (fdelete.exists()) {
            if (fdelete.delete()) {
                Log.d("PyTorchDemo", "file Deleted: " + filePath);
            } else {
                Log.e("PyTorchDemo", "file not Deleted: " + filePath);
            }
        } else {
            Log.e("PyTorchDemo", "file does not exist: " + filePath);
        }
    }

//    Utils:
    private void loadKeyFrameFeatures() {
        try {
            String jsonString = loadFile();
            if (jsonString.equals("")) {
                for (String keyFramePath : keyFramePaths) {
                    Uri keyFrameUri = Uri.parse(keyFramePath);
                    Bitmap bitmap = rotateBitmap(keyFrameUri);

                    bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, false);
                    Log.d("PyTorchDemo", "file " + keyFramePath + " width: " + bitmap.getWidth() + " height: " + bitmap.getHeight());

                    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                    final Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
                    float[] feature = outputTensor.getDataAsFloatArray();
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
            Log.e("PyTorchDemo", "Error reading keyframes", e);
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

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private Bitmap rotateBitmap(Uri uri) throws IOException {
        Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);

        InputStream in = getContentResolver().openInputStream(uri);
        ExifInterface exif = new ExifInterface(in);
        int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1);
        int orientationDegree = 0;
        if (orientation == ExifInterface.ORIENTATION_ROTATE_90) {
            orientationDegree = 90;
        } else if (orientation ==   ExifInterface.ORIENTATION_ROTATE_180) {
            orientationDegree = 180;
        } else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) {
            orientationDegree = 270;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(orientationDegree);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        return bitmap;
    }
}