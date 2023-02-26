using UnityEngine;
using System.Collections;
using UnityEditor;

[CustomEditor(typeof(GamePadDataCollector))]
public class ObjectBuilderEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        GamePadDataCollector dataCollection = (GamePadDataCollector)target;
        if (GUILayout.Button("Start Capture"))
        {
            if (Application.isPlaying)
            {
                dataCollection.StartCoroutine(dataCollection.StartDataCollection());
            }
            else
            {
                Debug.LogWarning("Cannot start data collection in editor mode.");
            }
        }
        
    }
}
