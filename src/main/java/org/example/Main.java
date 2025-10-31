package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;
import ai.djl.translate.Batchifier;

import java.io.IOException;
import java.nio.file.Paths;

// ------------------------------
// Custom input class
// ------------------------------
/**
 * MyInput represents the input for the Transformer Masked Autoencoder.
 * It contains all clusters except the missing one and the index of the missing cluster.
 */
class MyInput {
    float[][] data;   // shape: (num_clusters-1, 2)
    int maskIdx;      // Index of the missing cluster (0-based)

    public MyInput(float[][] data, int maskIdx) {
        this.data = data;
        this.maskIdx = maskIdx;
    }
}

// ------------------------------
// Main class
// ------------------------------
public class Main {

    public static void main(String[] args) {

        // ------------------------------
        // Define Translator
        // Input: MyInput
        // Output: float[2] containing predicted [avgWire, slope]
        // ------------------------------
        Translator<MyInput, float[]> myTranslator = new Translator<MyInput, float[]>() {

            @Override
            public NDList processInput(TranslatorContext ctx, MyInput input) throws Exception {
                NDManager manager = ctx.getNDManager();

                // ---- First input: float sequence, shape (1, num_clusters-1, 2) ----
                NDArray x = manager.create(input.data).reshape(1, input.data.length, 2);

                // ---- Second input: mask index, shape (1,) ----
                NDArray maskNd = manager.create(new int[]{input.maskIdx});

                return new NDList(x, maskNd);
            }

            @Override
            public float[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
                NDArray result = list.get(0); // shape (1, 2)
                return result.toFloatArray();  // [avgWire_pred, slope_pred]
            }

            @Override
            public Batchifier getBatchifier() {
                return null; // No batching for single prediction
            }
        };

        // ------------------------------
        // Define Criteria for loading the model
        // ------------------------------
        Criteria<MyInput, float[]> myModelCriteria = Criteria.builder()
                .setTypes(MyInput.class, float[].class)
                .optModelPath(Paths.get("nets/tmae_default.pt"))  // Path to PyTorch model
                .optEngine("PyTorch")
                .optTranslator(myTranslator)
                .optProgress(new ProgressBar())
                .build();

        // ------------------------------
        // Run inference
        // ------------------------------
        try (ZooModel<MyInput, float[]> model = myModelCriteria.loadModel();
             Predictor<MyInput, float[]> predictor = model.newPredictor()) {

            // Example input: 5 clusters, each has [avgWire, slope]
            float[][] inputArray = new float[][]{
                    {44.6000f, -0.1632f},
                    {43.3333f, -0.0946f},
                    {41.0000f, -0.1637f},
                    {38.8571f, -0.1837f},
                    {35.1667f, -0.2289f}
            };

            int missingIdx = 2;  // Index of cluster to predict

            MyInput input = new MyInput(inputArray, missingIdx);
            float[] output = predictor.predict(input);

            System.out.printf("Predicted cluster #%d -> avgWire = %.4f, slope = %.4f%n",
                    missingIdx, output[0], output[1]);

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
