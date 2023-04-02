using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Rewired;
using Rewired.ControllerExtensions;
using Unity.Barracuda;
using System;

public class AudioController : MonoBehaviour
{
    public int rewiredPlayerId = 0;

    private Player player;

    DualSenseExtension dualsense;

    public NNModel modelAsset;
    private Model m_RuntimeModel;

    private IWorker worker;
    Tensor modelInput;

    float[,] buffers;
    float[] controllerDataRMS;
    uint inputStreams = 10;
    uint bufferSize = 64; // Greatly affects frame rate, as some calculations are done every frame, so set it as low as feasible.
    float updateInterval = .5f; // In Seconds.

    //Prevents a coroutine (and more than one coroutine) from being triggered multiple times before finishing.
    bool isFading;
    int prediction;

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
        controllerDataRMS = new float[inputStreams];
    }

    // Start is called before the first frame update
    void Start()
    {
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RuntimeModel);
        StartCoroutine(UpdatePredictions());

        // Audio Setup
        for (int i = 0; i < audioSources.Length; i++)
        {
            audioSources[i].volume = 0.0f;
            audioSources[i].loop = true;
            //All the tracks have to be playing from scene start, or they won't sync properly.
            audioSources[i].Play();
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Continuously update RMS
        for (uint i = 0; i < bufferSize; i++)
        {
            buffers[0, i] = player.GetAxis("Move Horizontal");
            buffers[1, i] = player.GetAxis("Move Vertical");
            buffers[2, i] = player.GetAxis("Look Horizontal");
            buffers[3, i] = player.GetAxis("Look Vertical");
            buffers[4, i] = dualsense.GetGyroscopeValue().x;
            buffers[5, i] = dualsense.GetGyroscopeValue().y;
            buffers[6, i] = dualsense.GetGyroscopeValue().z;
            buffers[7, i] = dualsense.GetAccelerometerValue().x;
            buffers[8, i] = dualsense.GetAccelerometerValue().y;
            buffers[9, i] = dualsense.GetAccelerometerValue().z;

            // Calculate RMS for each buffer
            controllerDataRMS = RMS(buffers);
        }

        // Music control flow
        if (!isFading)
        {
            switch (prediction)
            {
                case 3:
                    // Activate Layers
                    if (audioSources[0].volume == 0.0f) StartCoroutine(FadeIn(audioSources[0], 1.0f, inDuration));
                    if (audioSources[1].volume == 0.0f) StartCoroutine(FadeIn(audioSources[1], 1.0f, inDuration));
                    if (audioSources[2].volume == 0.0f) StartCoroutine(FadeIn(audioSources[2], 1.0f, inDuration));
                    break;
                case 2:
                    // Activate Layers
                    if (audioSources[0].volume == 0.0f) StartCoroutine(FadeIn(audioSources[0], 1.0f, inDuration));
                    if (audioSources[1].volume == 0.0f) StartCoroutine(FadeIn(audioSources[1], 1.0f, inDuration));
                    // Deactivate Layers
                    if (audioSources[2].volume != 0.0f) StartCoroutine(FadeOut(audioSources[2], outDuration));
                    break;
                case 1:
                    // Activate Layers
                    if (audioSources[0].volume == 0.0f) StartCoroutine(FadeIn(audioSources[0], 1.0f, inDuration));
                    // Deactivate Layers
                    if (audioSources[2].volume != 0.0f) StartCoroutine(FadeOut(audioSources[2], outDuration));
                    if (audioSources[1].volume != 0.0f) StartCoroutine(FadeOut(audioSources[1], outDuration));
                    break;
                case 0:
                    // Fade out all music
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
        /* Using a coroutine to be able to choose how often to update predictions */

        // Always update predictions
        while (true)
        {
            // Inference
            modelInput = new Tensor(new int[2] { 1, (int)inputStreams }, controllerDataRMS);
            Tensor output = worker.Execute(modelInput).PeekOutput();

            float[] predictions = Softmax(output.AsFloats());

            
            Debug.Log("Prediction: " + predictions[0].ToString("F3") + 
                " " + predictions[1].ToString("F3") +
                " " + predictions[2].ToString("F3") +
                " " + predictions[3].ToString("F3") +
                " Selected: " + Argmax(predictions));

            // Update the prediction
            prediction = Argmax(predictions);

            modelInput?.Dispose();

            yield return new WaitForSecondsRealtime(updateInterval);
        }
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

    float[] Softmax(float[] x)
    {
        // e ^ (x - max(x)) / sum(e^(x - max(x))

        float[] softmaxOut = new float[x.Length];

        float max = 0;
        float sum = 0;

        for (var i = 0; i < x.Length; i++)
        {
            if (x[i] > max)
            {
                max = x[i];
            }

            sum += Mathf.Exp(x[i]);
        }

        for (int i = 0; i < x.Length; i++)
        {
            softmaxOut[i] = Mathf.Exp(x[i] - max) / sum;
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
        worker?.Dispose();
        modelInput?.Dispose();
    }
}
