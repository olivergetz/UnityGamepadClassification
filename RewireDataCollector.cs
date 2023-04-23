using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using Rewired;
using Rewired.ControllerExtensions;
using System.IO;

public class RewireDataCollector : MonoBehaviour
{
    public string fileName = "";
    string filepath;
    public int bufferSize;
    public int sampleRate;
    public int countdownTime;

    float sampleRateSeconds;

    string headers = "Timestamp, Left Stick X, Left Stick Y, Right Stick X, Right Stick Y, D-Pad Up, D-Pad Right," +
                        "D-Pad Down, D-Pad Left, Button North, Button East, Button South, Button West, L1, R1, L2, R2, L3, R3," +
                        "Gyroscope X, Gyroscope Y, Gyroscope Z, Accelerometer X, Accelerometer Y, Accelerometer Z," +
                        "Touch 1, Touch 2, Touch 1 Pos X, Touch 1 Pos Y, Touch 2 Pos X, Touch 2 Pos Y";


    public int rewiredPlayerId = 0;
    
    private Player player;

    DualSenseExtension dualsense;

    [System.Serializable]
    public class GamepadData
    {
        public float[] timestamp;
        public float[] leftStickX;
        public float[] leftStickY;
        public float[] rightStickX;
        public float[] rightStickY;
        public int[] dpadUp;
        public int[] dpadRight;
        public int[] dpadDown;
        public int[] dpadLeft;
        public int[] buttonNorth;
        public int[] buttonEast;
        public int[] buttonSouth;
        public int[] buttonWest;
        public int[] leftShoulder;
        public int[] rightShoulder;
        public int[] leftTrigger;
        public int[] rightTrigger;
        public int[] leftStick;
        public int[] rightStick;
        public float[] gyroscopeX;
        public float[] gyroscopeY;
        public float[] gyroscopeZ;
        public float[] accelerometerX;
        public float[] accelerometerY;
        public float[] accelerometerZ;
        public int[] touch1;
        public int[] touch2;
        public float[] touch1PositionX;
        public float[] touch1PositionY;
        public float[] touch2PositionX;
        public float[] touch2PositionY;

        public GamepadData(int bufferSize)
        {
            timestamp = new float[bufferSize];
            leftStickX = new float[bufferSize];
            leftStickY = new float[bufferSize];
            rightStickX = new float[bufferSize];
            rightStickY = new float[bufferSize];
            dpadUp = new int[bufferSize];
            dpadRight = new int[bufferSize];
            dpadDown = new int[bufferSize];
            dpadLeft = new int[bufferSize];
            buttonNorth = new int[bufferSize];
            buttonEast = new int[bufferSize];
            buttonSouth = new int[bufferSize];
            buttonWest = new int[bufferSize];
            leftShoulder = new int[bufferSize];
            rightShoulder = new int[bufferSize];
            leftTrigger = new int[bufferSize];
            rightTrigger = new int[bufferSize];
            leftStick = new int[bufferSize];
            rightStick = new int[bufferSize];
            gyroscopeX = new float[bufferSize];
            gyroscopeY = new float[bufferSize];
            gyroscopeZ = new float[bufferSize];
            accelerometerX = new float[bufferSize];
            accelerometerY = new float[bufferSize];
            accelerometerZ = new float[bufferSize];
            touch1 = new int[bufferSize]; ;
            touch2 = new int[bufferSize]; ;
            touch1PositionX = new float[bufferSize];
            touch1PositionY = new float[bufferSize];
            touch2PositionX = new float[bufferSize];
            touch2PositionY = new float[bufferSize];
        }

    }

    GamepadData gamepadData;

    void Awake()
    {
        ConnectController();
        Application.runInBackground = true;
    }
    private void Start()
    {
        gamepadData = new GamepadData(bufferSize);
        sampleRateSeconds = 1.0f / sampleRate;
        filepath = Application.dataPath + "/Scripts/CollectedData/" + fileName + ".csv";
    }

    void Update()
    {
        //Debug.Log(player.GetAxis("Move Horizontal"));
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


    public void ToCSV()
    {
        if (gamepadData != null)
        {
            TextWriter tw = new StreamWriter(filepath, false);
            tw.WriteLine(headers);
            tw.Close();

            tw = new StreamWriter(filepath, true);

            for (int i = 0; i < bufferSize; i++)
            {
                string percent = (i / (float)bufferSize).ToString("F2");
                Debug.Log("Saving Data... " + i + "/" + bufferSize + " " + percent + "%");
                tw.WriteLine(gamepadData.timestamp[i] + "," + gamepadData.leftStickX[i] + "," + gamepadData.leftStickY[i] + "," + gamepadData.rightStickX[i] + "," + gamepadData.rightStickY[i] + ","
                    + gamepadData.dpadUp[i] + "," + gamepadData.dpadRight[i] + "," + gamepadData.dpadDown[i] + "," + gamepadData.dpadLeft[i] + ","
                    + gamepadData.buttonNorth[i] + "," + gamepadData.buttonEast[i] + "," + gamepadData.buttonSouth[i] + "," + gamepadData.buttonWest[i] + ","
                    + gamepadData.leftShoulder[i] + "," + gamepadData.rightShoulder[i] + "," + gamepadData.leftTrigger[i] + "," + gamepadData.rightTrigger[i] + ","
                    + gamepadData.leftStick[i] + "," + gamepadData.rightStick[i] + "," + gamepadData.gyroscopeX[i] + "," + gamepadData.gyroscopeY[i] + ","
                    + gamepadData.gyroscopeZ[i] + "," + gamepadData.accelerometerX[i] + "," + gamepadData.accelerometerY[i] + "," + gamepadData.accelerometerZ[i] + ","
                    + gamepadData.touch1[i] + "," + gamepadData.touch2[i] + "," + gamepadData.touch1PositionX[i] + "," + gamepadData.touch1PositionY[i] + "," + gamepadData.touch2PositionX[i] + ","
                    + gamepadData.touch2PositionY[i]);

            }

            Debug.Log("Data Collected!");

        }

    }

    public IEnumerator StartDataCollection()
    {
        // Count Down
        for (int i = 0; i < countdownTime; i++)
        {
            Debug.Log("Starting capture in: " + (countdownTime - i));
            yield return new WaitForSecondsRealtime(1);
        }

        Debug.Log("Capturing Data");
        float elapsedTime = 0f;
        for (int i = 0; i < bufferSize; i++)
        {
            string percent = (i / (float)bufferSize).ToString("F2");
            Debug.Log("Collecting Data... " + i + "/" + bufferSize + " " + percent + "%");
            gamepadData.timestamp[i] = elapsedTime;
            gamepadData.leftStickX[i] = player.GetAxis("Move Horizontal");
            gamepadData.leftStickY[i] = player.GetAxis("Move Vertical");
            gamepadData.rightStickX[i] = player.GetAxis("Look Horizontal");
            gamepadData.rightStickY[i] = player.GetAxis("Look Vertical");
            gamepadData.dpadUp[i] = Convert.ToInt32(player.GetButton("D-Pad Up"));
            gamepadData.dpadRight[i] = Convert.ToInt32(player.GetButton("D-Pad Right"));
            gamepadData.dpadDown[i] = Convert.ToInt32(player.GetButton("D-Pad Down"));
            gamepadData.dpadLeft[i] = Convert.ToInt32(player.GetButton("D-Pad Left"));
            gamepadData.buttonNorth[i] = Convert.ToInt32(player.GetButton("Button North"));
            gamepadData.buttonEast[i] = Convert.ToInt32(player.GetButton("Button East"));
            gamepadData.buttonSouth[i] = Convert.ToInt32(player.GetButton("Button South"));
            gamepadData.buttonWest[i] = Convert.ToInt32(player.GetButton("Button West"));
            gamepadData.leftShoulder[i] = Convert.ToInt32(player.GetButton("Left Shoulder"));
            gamepadData.rightShoulder[i] = Convert.ToInt32(player.GetButton("Right Shoulder"));
            gamepadData.leftTrigger[i] = Convert.ToInt32(player.GetButton("Left Trigger"));
            gamepadData.rightTrigger[i] = Convert.ToInt32(player.GetButton("Right Trigger"));
            gamepadData.leftStick[i] = Convert.ToInt32(player.GetButton("Left Stick Button"));
            gamepadData.rightStick[i] = Convert.ToInt32(player.GetButton("Right Stick Button"));
            gamepadData.gyroscopeX[i] = dualsense.GetGyroscopeValue().x;
            gamepadData.gyroscopeY[i] = dualsense.GetGyroscopeValue().y;
            gamepadData.gyroscopeZ[i] = dualsense.GetGyroscopeValue().z;
            gamepadData.accelerometerX[i] = dualsense.GetAccelerometerValue().x;
            gamepadData.accelerometerY[i] = dualsense.GetAccelerometerValue().y;
            gamepadData.accelerometerZ[i] = dualsense.GetAccelerometerValue().z;
            Vector2 touch1Pos, touch2Pos;
            gamepadData.touch1[i] = Convert.ToInt32(dualsense.GetTouchPosition(0, out touch1Pos));
            gamepadData.touch1[i] = Convert.ToInt32(dualsense.GetTouchPosition(0, out touch2Pos));
            gamepadData.touch1PositionX[i] = touch1Pos.x;
            gamepadData.touch1PositionY[i] = touch1Pos.y;
            gamepadData.touch2PositionX[i] = touch1Pos.x;
            gamepadData.touch2PositionY[i] = touch1Pos.y;

            elapsedTime += sampleRateSeconds;

            yield return new WaitForSecondsRealtime(sampleRateSeconds);
        }

        ToCSV();
    }
}
