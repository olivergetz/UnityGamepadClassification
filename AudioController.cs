using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Rewired;
using Rewired.ControllerExtensions;
using Unity.Barracuda;
using System;
using System.Linq;

public class AudioController : MonoBehaviour
{
    [Space(10)]
    [Header("Controller Selection")]
    public int rewiredPlayerId = 0;

    private Player player;

    DualSenseExtension dualsense;

    [Space(10)]
    [Header("Models")]
    public NNModel[] modelAssets;
    private Model[] m_RuntimeModels;
    private IWorker[] workers;
    Tensor modelInput;
    Tensor[] outputs;

    public enum FusionMethod
    {
        MajorityVote,
        AverageProbabilities
    }

    public FusionMethod fusionMethod = FusionMethod.AverageProbabilities;

    float[,] buffers;
    float[] controllerDataFeats; // All features, interleaved like (a1, a2, a3, b1, b2, b3...)
    uint inputStreams = 15;
    int numFeatures = 3;

    [Space(10)]
    [Header("Included Features")]
    // Inlcude or exclude features during testing
    public bool useMean = false;
    public bool useVariance = false;
    public bool useRMS = false;
    bool[] featuresToUse;

    [Space(10)]
    [Header("Inference Settings")]
        // BufferSize is used to calculate features like RMS.
    // Greatly affects frame rate, as some calculations are done every frame, so set it as low as feasible.
    public uint bufferSize = 256; 
    public float updateInterval = 1f; // How often to update features, in Seconds.

    //Prevents a coroutine (and more than one coroutine) from being triggered multiple times before finishing.
    bool isFading;
    int prediction;
    float avgFraction; // Used to calculate mean repeatedly when doing AverageVote fusion.

    [Space(10)]
    [Header("Audio")]
    public AudioSource[] audioSources;

    [Range(0.1f, 10.0f)] //Min. range must be 0.1f or greater.
    [SerializeField]
    private float inDuration = 5.0f;
    [Range(0.1f, 10.0f)]
    [SerializeField]
    private float outDuration = 5.0f;

    void Awake()
    {
        ConnectController();
        Application.runInBackground = true;

        buffers = new float[inputStreams, bufferSize];
        controllerDataFeats = new float[inputStreams*numFeatures];

        avgFraction = (1 / modelAssets.Length);

        // Store feature states as a single array, for counting Trues and interleaving features.
        featuresToUse = new bool[] { useMean, useVariance, useRMS };
        // Get number of features in use.
        numFeatures = featuresToUse.Where(c => c).Count();
    }

    // Start is called before the first frame update
    void Start()
    {
        // Load models
        m_RuntimeModels = new Model[modelAssets.Length];
        workers = new IWorker[modelAssets.Length];
        outputs = new Tensor[modelAssets.Length];

        // Create workers
        for (int i = 0; i < m_RuntimeModels.Length; i++)
        {
            m_RuntimeModels[i] = ModelLoader.Load(modelAssets[i]);
            workers[i] = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RuntimeModels[i]);
        }

        // Audio Setup
        for (int i = 0; i < audioSources.Length; i++)
        {
            audioSources[i].volume = 0.0f;
            audioSources[i].loop = true;
            //All the tracks have to be playing from scene start, or they won't sync properly.
            audioSources[i].Play();
        }

        StartCoroutine(UpdatePredictions());
    }

    // Update is called once per frame
    void Update()
    {

        // Music control flow
        if (!isFading)
        {
            switch (prediction)
            {
                case 0:
                    // Activate Layers - High Activity
                    if (audioSources[0].volume == 0.0f) StartCoroutine(FadeIn(audioSources[0], 1.0f, inDuration));
                    if (audioSources[1].volume == 0.0f) StartCoroutine(FadeIn(audioSources[1], 1.0f, inDuration));
                    if (audioSources[2].volume == 0.0f) StartCoroutine(FadeIn(audioSources[2], 1.0f, inDuration));
                    break;
                case 3:
                    // Activate Layers - Medium Activity
                    if (audioSources[0].volume == 0.0f) StartCoroutine(FadeIn(audioSources[0], 1.0f, inDuration));
                    if (audioSources[1].volume == 0.0f) StartCoroutine(FadeIn(audioSources[1], 1.0f, inDuration));
                    // Deactivate Layers
                    if (audioSources[2].volume != 0.0f) StartCoroutine(FadeOut(audioSources[2], outDuration));
                    break;
                case 2:
                    // Activate Layers - Low Activity
                    if (audioSources[0].volume == 0.0f) StartCoroutine(FadeIn(audioSources[0], 1.0f, inDuration));
                    // Deactivate Layers
                    if (audioSources[2].volume != 0.0f) StartCoroutine(FadeOut(audioSources[2], outDuration));
                    if (audioSources[1].volume != 0.0f) StartCoroutine(FadeOut(audioSources[1], outDuration));
                    break;
                case 1:
                    // Fade out all music - Idle
                    for (int i = 0; i < audioSources.Length; i++)
                    {
                        if (audioSources[i].volume != 0.0f) StartCoroutine(FadeOut(audioSources[i], outDuration));
                    }
                    break;
                default:
                    // Fade out all music
                    for (int i = 0; i < audioSources.Length; i++)
                    {
                        if (audioSources[i].volume != 0.0f) StartCoroutine(FadeOut(audioSources[i], outDuration));
                    }
                    break;
            }
        }
    }

    void UpdateBuffers()
    {
        // Update buffers and calculate features
        for (uint i = 0; i < bufferSize; i++)
        {
            buffers[0, i] = player.GetAxis("Move Horizontal");
            buffers[1, i] = player.GetAxis("Move Vertical");
            buffers[2, i] = player.GetAxis("Look Horizontal");
            buffers[3, i] = player.GetAxis("Look Vertical");
            buffers[4, i] = Convert.ToInt32(player.GetButton("Button South"));
            buffers[5, i] = Convert.ToInt32(player.GetButton("Button West"));
            buffers[6, i] = Convert.ToInt32(player.GetButton("Left Shoulder"));
            buffers[7, i] = Convert.ToInt32(player.GetButton("Right Shoulder"));
            buffers[8, i] = Convert.ToInt32(player.GetButton("Right Trigger"));
            buffers[9, i] = dualsense.GetGyroscopeValue().x;
            buffers[10, i] = dualsense.GetGyroscopeValue().y;
            buffers[11, i] = dualsense.GetGyroscopeValue().z;
            buffers[12, i] = dualsense.GetAccelerometerValue().x;
            buffers[13, i] = dualsense.GetAccelerometerValue().y;
            buffers[14, i] = dualsense.GetAccelerometerValue().z;
        }

        // Calculate Mean, Variance, and/or RMS for each buffer
        controllerDataFeats = CalculateFeatures(buffers, useMean, useVariance, useRMS);
    }

    // Use this when there are 3 different features.
    float[] InterleaveFloats(float[] x, float[] y, float[] z)
    {
        float[] interleavedFloats = new float[x.Length * 3];
        
        for (uint i = 0; i < x.Length; i++)
        {
            for (uint j = 0; j < 3; j++)
            {
                // Modulo is used to update only one variable per iteration of j.
                uint idx = j % 3;
                if (idx == 0) interleavedFloats[3 * i + idx] = x[i]; // 0, 3, 6, 9
                if (idx == 1) interleavedFloats[3 * i + idx] = y[i]; // 1, 4, 7, 10
                if (idx == 2) interleavedFloats[3 * i + idx] = z[i]; // 2, 5, 8, 11
            }
        }

        return interleavedFloats;
    }

    // Use this when there are 2 different features.
    float[] InterleaveFloats(float[] x, float[] y)
    {
        float[] interleavedFloats = new float[x.Length * 3];

        for (uint i = 0; i < x.Length; i++)
        {
            for (uint j = 0; j < 2; j++)
            {
                // Modulo is used to update only one variable per iteration of j.
                uint idx = j % 2;
                if (idx == 0) interleavedFloats[2 * i + idx] = x[i]; // 0, 2, 4, 6
                if (idx == 1) interleavedFloats[2 * i + idx] = y[i]; // 1, 3, 5, 7
            }
        }

        return interleavedFloats;
    }

    IEnumerator FadeIn(AudioSource source, float finish, float duration)
    {

        isFading = true;

        for (float t = 0.0f; t < duration; t += Time.deltaTime)
        {
            float normalizedTime = t / duration;
            source.volume = Mathf.Lerp(source.volume, finish, normalizedTime);
            yield return null;
        }

        isFading = false;
        StopCoroutine("FadeIn");

    }

    IEnumerator FadeOut(AudioSource source, float duration)
    {

        isFading = true;

        for (float t = 0.0f; t < duration; t += Time.deltaTime)
        {
            float normalizedTime = t / duration;
            source.volume = Mathf.Lerp(source.volume, 0.0f, normalizedTime);
            yield return null;
        }

        source.volume = 0.0f;

        isFading = false;
        StopCoroutine("FadeOut");

    }

    IEnumerator UpdatePredictions()
    {
        /*
         * We use a coroutine to control how often to update predictions and buffers.
         * 
         * Model input must be a tensor of shape (1, inputStreams*n_features)
         * If you get this error:
         * 
         * AssertionException: Assertion failure. Values are not equal.
         * Expected: n_expected_input == n_supplied_input
         *  
         * Ensure your selected features match the features the model was trained on.
         */

        // Always update predictions
        while (true)
        {
            // We update buffers here to only do so according to our update interval.
            UpdateBuffers();

            // Convert controller input to tensor
            modelInput = new Tensor(new int[2] { 1, (int)(inputStreams*numFeatures) }, controllerDataFeats);

            // Storage for predictions
            float[][] predictions = new float[modelAssets.Length][];

            // Make predictions
            for (int i = 0; i < workers.Length; i++)
            {
                outputs[i] = workers[i].Execute(modelInput).PeekOutput();
                predictions[i] = Softmax(outputs[i].AsFloats());
            }

            // Fusion predictions, if any.
            if (modelAssets.Length == 1)
            {
                // Update the prediction
                prediction = outputs[0].ArgMax()[0];

                Debug.Log("Prediction: " + predictions[0][0].ToString("F3") +
                " " + predictions[0][1].ToString("F3") +
                " " + predictions[0][2].ToString("F3") +
                " " + predictions[0][3].ToString("F3") +
                " Selected: " + prediction);
            } 
            else if (fusionMethod == FusionMethod.MajorityVote)
            {
                // Contains the number of votes for each class.
                int[] votes = new int[predictions[0].Length];

                // Count votes
                for (int i = 0; i < predictions.Length; i++)
                {
                    int vote = outputs[i].ArgMax()[0];
                    votes[vote] += 1; 
                }

                // Make decision
                // Update the prediction
                prediction = Argmax(votes);

                Debug.Log("Prediction: " + predictions[0][0].ToString("F3") +
                " " + predictions[0][1].ToString("F3") +
                " " + predictions[0][2].ToString("F3") +
                " " + predictions[0][3].ToString("F3") +
                " Selected: " + prediction);

            }
            else if (fusionMethod == FusionMethod.AverageProbabilities)
            {
                // Container for final predictions, of size n classes.
                float[] finalPrediction = new float[predictions[0].Length];
                // For each class - predictions[0].Length should be equal to the number of classes
                for (int i = 0; i < predictions[0].Length; i++)
                {
                    float sum = 0;

                    // Calculate the mean along each element - predictions.Length should be the number of predictions made, i.e. 3 if there are 3 models.
                    for (int j = 0; j < predictions.Length; j++)
                    {
                        // Explained: finalPrediction[currentClass] = predictions[value][currentClass]
                        
                        sum += predictions[j][i];
                        finalPrediction[i] = avgFraction * sum;
                    }
                }

                // Update the prediction
                prediction = Argmax(finalPrediction);

                Debug.Log("Prediction: " + finalPrediction[0].ToString("F3") +
                " " + finalPrediction[1].ToString("F3") +
                " " + finalPrediction[2].ToString("F3") +
                " " + finalPrediction[3].ToString("F3") +
                " Selected: " + prediction);

            }

            modelInput?.Dispose();

            yield return new WaitForSecondsRealtime(updateInterval);
        }
    }

    // Option to supply mean, so it's not calculated twice.
    float Variance(float[] x, float mean)
    {
        float variance = 0;

        for (var i = 0; i < x.Length; i++)
        {
            variance += Mathf.Pow(x[i] - mean, 2.0f);
        }

        return variance;
    }

    float Variance(float[] x)
    {
        float mean = Mean(x);
        float variance = 0;

        for (var i = 0; i < x.Length; i++)
        {
            variance += Mathf.Pow(x[i] - mean, 2.0f);
        }

        return variance;
    }

    // Features must be supplied to the model in the format (Mean, Var, RMS) for each input stream.
    // If FPS becomes an issue, restructure the data before training the model.
    float[] CalculateFeatures(float[,] x, bool calculateMean = false, bool calculateVar = false, bool calculateRMS = false)
    {
        // I can't find a way to guard against 3 Falses, so I guess I'll just let the program die.
        // Microsoft says: don't throw exceptions from your own code.
        bool[] featuresToUse = new bool[] { calculateMean, calculateVar, calculateRMS };
        int numFeatures = featuresToUse.Where(c => c).Count();

        if (numFeatures == 0)
        {
            Debug.LogWarning("No features are selected. Select at least 1.");
            return new float[0];
        }

        int inputs = x.GetLength(0);
        int bufferSize = x.GetLength(1);
        float[] buffer = new float[inputs];   // The buffer to compute features from.
        float[] outRMS = new float[inputs];
        float[] outMean = new float[inputs];
        float[] outVar = new float[inputs];

        for (int i = 0; i < inputs; i++)
        {
            for (int j = 0; j < bufferSize; j++)
            {
                // Copy values j from the current buffer i to a new array so the features can be calculated for this set of values.
                buffer[i] = x[i, j];
            }

            // Calculate Mean if either Mean or Variance is marked as True.
            if (featuresToUse[0] || featuresToUse[1]) outMean[i] = Mean(buffer);
            // Calculate Variance
            if (featuresToUse[1]) outVar[i] = Variance(buffer, outMean[i]);

            //(outMean[i], outVar[i]) = MeanAndVariance(buffer);
            // Calculate RMS for the current buffer
            if (featuresToUse[2]) outRMS[i] = RMS(buffer);
            
        }

        // Select features and interleave
        if (featuresToUse[0])
        {
            if (featuresToUse[1])
            {
                if (featuresToUse[2]) return InterleaveFloats(outMean, outVar, outRMS);
                else return InterleaveFloats(outMean, outVar);
            }
            else if (featuresToUse[2]) return InterleaveFloats(outMean, outRMS);
            else return outMean;
        }
        else if(featuresToUse[1])
        {
            if (featuresToUse[2]) return InterleaveFloats(outVar, outRMS);
            else return outVar;
        }
        else if (featuresToUse[2]) return outRMS;

        return new float[0];
    }

    float Mean(float[] x)
    {
        float sum = 0;

        for (var i = 0; i < x.Length; i++)
        {
            sum += x[i];
        }

        return sum / x.Length;
    }

    float RMS(float[] x)
    {
        /* RMS over a buffer */
        float sum = 0;

        foreach (float value in x)
        {
            sum += Mathf.Pow(value, 2);
        }

        return Mathf.Sqrt(sum / x.Length);
    }

    float[] RMS(float[,] x)
    {
        /* RMS over multiple buffers */
        int inputs = x.GetLength(0);
        int bufferSize = x.GetLength(1);
        float[] buffer = new float[inputs];   // Output RMS values, one for each controller input
        float[] outRMS = new float[inputs];

        for (int i = 0; i < inputs; i++)
        {
            for (int j = 0; j < bufferSize; j++)
            {
                // Copy values j from the current buffer i to a new array so the RMS can be calculated for this set of values.
                buffer[i] = x[i, j];
            }

            // Calculate RMS for the current buffer
            outRMS[i] = RMS(buffer);

        }

        return outRMS;
    }

    // Used when doing average probability fusion.
    int Argmax(float[] x)
    {
        int outmax = 0;
        float highestValue = 0;

        // Iterate over input to find the highest number
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] > highestValue)
            {
                outmax = i;
                highestValue = x[i];
            }
        }

        return outmax;
    }

    // Used when doing average probability fusion.
    int Argmax(int[] x)
    {
        int outmax = 0;
        int highestValue = 0;

        // Iterate over input to find the highest number
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] > highestValue)
            {
                outmax = i;
                highestValue = x[i];
            }
        }

        return outmax;
    }

    float[] Softmax(float[] z)
    {
        // e ^ (x - max(x)) / sum(e^(x - max(x))

        float[] softmaxOut = new float[z.Length];
        float[] z_exp = new float[z.Length];

        float sum_z_exp = 0;

        for (var i = 0; i < z.Length; i++)
        {
            z_exp[i] = Mathf.Exp(z[i]);
            sum_z_exp += z_exp[i];
        }

        for (int i = 0; i < z.Length; i++)
        {
            softmaxOut[i] = z_exp[i] / sum_z_exp;
        }

        return softmaxOut;

    }

    void ConnectController()
    {
        // Get the Rewired Player object for this player and keep it for the duration of the character's lifetime
        player = ReInput.players.GetPlayer(rewiredPlayerId);

        // Loop through all Joysticks assigned to this Player and find the first dual sense controller
        foreach (Joystick joystick in player.controllers.Joysticks)
        {
            dualsense = joystick.GetExtension<DualSenseExtension>();
        }
    }

    private void OnDestroy()
    {
        for (int i = 0; i < workers.Length; i++)
        {
            workers[i]?.Dispose();
        }
        
        modelInput?.Dispose();
    }
}
