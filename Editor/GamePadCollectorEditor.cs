using UnityEngine;
using System.Collections;
using UnityEditor;

[CustomEditor(typeof(RewireDataCollector))]
public class ObjectBuilderEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        RewireDataCollector dataCollection = (RewireDataCollector)target;
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
