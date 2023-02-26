using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class GamePadData
{
    public int buffer_size;

    public GamePadData(int buffer_size)
    {
        float[] leftStickX = new float[buffer_size];
        float[] leftStickY = new float[buffer_size];
        float[] rightStickX = new float[buffer_size];
        float[] rightStickY = new float[buffer_size];
        int[] dpadUp = new int[buffer_size];
        int[] dpadRight = new int[buffer_size];
        int[] dpadDown = new int[buffer_size];
        int[] dpadLeft = new int[buffer_size];
        int[] buttonNorth = new int[buffer_size];
        int[] buttonEast = new int[buffer_size];
        int[] buttonSouth = new int[buffer_size];
        int[] buttonWest = new int[buffer_size];
        float[] gyroscopeX = new float[buffer_size];
        float[] gyroscopeY = new float[buffer_size];
        float[] gyroscopeZ = new float[buffer_size];
        float[] accelerometerX = new float[buffer_size];
        float[] accelerometerY = new float[buffer_size];
        float[] accelerometerZ = new float[buffer_size];
    }
}
