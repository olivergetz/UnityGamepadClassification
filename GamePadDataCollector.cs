using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.DualShock;
using System.IO;
using Gyroscope = UnityEngine.InputSystem.Gyroscope;

public class GamePadDataCollector : MonoBehaviour
{
    public string fileName = "";
    string filepath;
    public int buffer_size;
    public int sample_rate;

    float sample_rate_seconds;

    string headers = "Left Stick X, LeftStickY, Right Stick X, Right Stick Y, D-Pad Up, D-Pad Right," +
                        "D-Pad Down, D-Pad Left, Button North, Button East, Button South, L1, R1, L2, R2, L3, R3" + 
                        "Gyroscope X, Gyroscope Y, Gyroscope Z, Accelerometer X, Accelerometer Y, Accelerometer Z";

    Gamepad dualSenseGamepad = new Gamepad();

    [System.Serializable]
    public class GamepadData
    {

        public float[] leftStickX;
        public float[] leftStickY;
        public float[] rightStickX;
        public float[] rightStickY;
        public float[] dpadUp;
        public float[] dpadRight;
        public float[] dpadDown;
        public float[] dpadLeft;
        public float[] buttonNorth;
        public float[] buttonEast;
        public float[] buttonSouth;
        public float[] buttonWest;
        public float[] leftShoulder;
        public float[] rightShoulder;
        public float[] leftTrigger;
        public float[] rightTrigger;
        public float[] leftStick;
        public float[] rightStick;
        public float[] gyroscopeX;
        public float[] gyroscopeY;
        public float[] gyroscopeZ;
        public float[] accelerometerX;
        public float[] accelerometerY;
        public float[] accelerometerZ;

        public GamepadData(int buffer_size)
        {
            leftStickX = new float[buffer_size];
            leftStickY = new float[buffer_size];
            rightStickX = new float[buffer_size];
            rightStickY = new float[buffer_size];
            dpadUp = new float[buffer_size];
            dpadRight = new float[buffer_size];
            dpadDown = new float[buffer_size];
            dpadLeft = new float[buffer_size];
            buttonNorth = new float[buffer_size];
            buttonEast = new float[buffer_size];
            buttonSouth = new float[buffer_size];
            buttonWest = new float[buffer_size];
            leftShoulder = new float[buffer_size];
            rightShoulder = new float[buffer_size];
            leftTrigger = new float[buffer_size];
            rightTrigger = new float[buffer_size];
            leftStick = new float[buffer_size];
            rightStick = new float[buffer_size];
            gyroscopeX = new float[buffer_size];
            gyroscopeY = new float[buffer_size];
            gyroscopeZ = new float[buffer_size];
            accelerometerX = new float[buffer_size];
            accelerometerY = new float[buffer_size];
            accelerometerZ = new float[buffer_size];
        }

        
    }


    GamepadData gamepadData;

    // Start is called before the first frame update
    void Start()
    {
        
        foreach (InputDevice device in InputSystem.devices)
        {
            if (device.name == "DualSenseGamepadHID")
            {
                dualSenseGamepad = (DualSenseGamepadHID)device;
            }

        }

        Debug.Log(dualSenseGamepad);
        sample_rate_seconds = 1/sample_rate;

        //InputSystem.EnableDevice(UnityEngine.InputSystem.Gyroscope.current);
        //InputSystem.EnableDevice(UnityEngine.InputSystem.Accelerometer.current);

        //dataCollector = new DataCollector(buffer_size);
        gamepadData = new GamepadData(buffer_size);
        gamepadData.leftStickX[1] = dualSenseGamepad.leftStick.x.ReadValue();
        filepath = Application.dataPath + "/" + fileName + ".csv";
        //ToCSV();
    }

    // Update is called once per frame
    void Update()
    {

        // Start Collection

        //Debug.Log(Gamepad.current.leftStick.x.ReadValue());
        // Check whether the X button on the current gamepad is pressed.
        //Debug.Log(Gamepad.current.leftStick.x.ReadValue());
        //Debug.Log(dualSenseGamepad.leftStick.y);
    }

    public void ToCSV()
    {
        if (gamepadData != null)
        {
            TextWriter tw = new StreamWriter(filepath, false);
            tw.WriteLine(headers);
            tw.Close();

            tw = new StreamWriter(filepath, true);

            for (int i = 0; i < buffer_size; i++)
            {
                tw.WriteLine(gamepadData.leftStickX[i] + "," + gamepadData.leftStickY[i] + "," + gamepadData.rightStickX[i] + "," + gamepadData.rightStickY[i] + ","
                    + gamepadData.dpadUp[i] + "," + gamepadData.dpadRight[i] + "," + gamepadData.dpadDown[i] + "," + gamepadData.dpadLeft[i] + ","
                    + gamepadData.buttonNorth[i] + "," + gamepadData.buttonEast[i] + "," + gamepadData.buttonSouth[i] + "," + gamepadData.buttonWest[i] + ","
                    + gamepadData.leftShoulder[i] + "," + gamepadData.rightShoulder[i] + "," + gamepadData.leftTrigger[i] + "," + gamepadData.rightTrigger[i] + ","
                    + gamepadData.leftStick[i] + "," + gamepadData.rightStick[i] + "," + gamepadData.gyroscopeX[i] + "," + gamepadData.gyroscopeY[i] + ","
                    + gamepadData.gyroscopeZ[i] + "," + gamepadData.accelerometerX[i] + "," + gamepadData.accelerometerY[i] + "," + gamepadData.accelerometerZ[i] + ",");
                
            }
        }
        
    }

    public IEnumerator StartDataCollection()
    {

        for (int i = 0; i < buffer_size; i++)
        {
            Debug.Log("Collecting Data " + i + "/" + buffer_size + " " + 1/buffer_size + "%");

            gamepadData.leftStickX[i] = Gamepad.current.leftStick.x.ReadValue();
            gamepadData.leftStickY[i] = Gamepad.current.leftStick.y.ReadValue();
            gamepadData.rightStickX[i] = Gamepad.current.rightStick.x.ReadValue();
            gamepadData.rightStickY[i] = Gamepad.current.rightStick.x.ReadValue();
            gamepadData.dpadUp[i] = Gamepad.current.dpad.up.ReadValue();
            gamepadData.dpadRight[i] = Gamepad.current.dpad.right.ReadValue();
            gamepadData.dpadDown[i] = Gamepad.current.dpad.down.ReadValue();
            gamepadData.dpadLeft[i] = Gamepad.current.dpad.left.ReadValue();
            gamepadData.buttonNorth[i] = Gamepad.current.buttonNorth.ReadValue();
            gamepadData.buttonEast[i] = Gamepad.current.buttonEast.ReadValue();
            gamepadData.buttonSouth[i] = Gamepad.current.buttonSouth.ReadValue();
            gamepadData.buttonWest[i] = Gamepad.current.buttonWest.ReadValue();
            gamepadData.leftShoulder[i] = Gamepad.current.leftShoulder.ReadValue();
            gamepadData.rightShoulder[i] = Gamepad.current.rightShoulder.ReadValue();
            gamepadData.leftTrigger[i] = Gamepad.current.leftTrigger.ReadValue();
            gamepadData.rightTrigger[i] = Gamepad.current.rightTrigger.ReadValue();
            gamepadData.leftStick[i] = Gamepad.current.leftStickButton.ReadValue();
            gamepadData.rightStick[i] = Gamepad.current.rightStickButton.ReadValue();
            //gamepadData.gyroscopeX[i] = Gyroscope.current.;
            //gamepadData.gyroscopeY[i]
            //gamepadData.gyroscopeZ[i]
            //gamepadData.accelerometerX[i]
            //gamepadData.accelerometerY[i]
            //gamepadData.accelerometerZ[i]

            yield return new WaitForSeconds(sample_rate_seconds);
        }

        ToCSV();
    }
}
