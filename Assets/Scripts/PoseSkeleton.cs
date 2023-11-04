using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class PoseSkeleton
{

    // The list of key point GameObjects that make up the pose skeleton
    public Transform[] keypoints;

    // The GameObjects that contain data for the lines between key points
    private GameObject[] lines;
    

    private Material sharedMaterial;

    // The names of the body parts that will be detected by the PoseNet model
    private static string[] partNames = new string[]{
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };

    private static int NUM_KEYPOINTS = partNames.Length;

    // The pairs of key points that should be connected on a body
    private Tuple<int, int>[] jointPairs = new Tuple<int, int>[]{
    // Nose to Left Eye
    Tuple.Create(0, 1),
    // Nose to Right Eye
    Tuple.Create(0, 2),
    // Left Eye to Left Ear
    Tuple.Create(1, 3),
    // Right Eye to Right Ear
    Tuple.Create(2, 4),
    // Left Shoulder to Right Shoulder
    Tuple.Create(5, 6),
    // Left Shoulder to Left Hip
    Tuple.Create(5, 11),
    // Right Shoulder to Right Hip
    Tuple.Create(6, 12),
    // Left Shoulder to Right Hip
    Tuple.Create(5, 12),
    // Rigth Shoulder to Left Hip
    Tuple.Create(6, 11),
    // Left Hip to Right Hip
    Tuple.Create(11, 12),
    // Left Shoulder to Left Elbow
    Tuple.Create(5, 7),
    // Left Elbow to Left Wrist
    Tuple.Create(7, 9), 
    // Right Shoulder to Right Elbow
    Tuple.Create(6, 8),
    // Right Elbow to Right Wrist
    Tuple.Create(8, 10),
    // Left Hip to Left Knee
    Tuple.Create(11, 13), 
    // Left Knee to Left Ankle
    Tuple.Create(13, 15),
    // Right Hip to Right Knee
    Tuple.Create(12, 14), 
    // Right Knee to Right Ankle
    Tuple.Create(14, 16)
    };

    // Colors for the skeleton lines
    private Color[] colors = new Color[] {
    // Head
    Color.magenta, Color.magenta, Color.magenta, Color.magenta,
    // Torso
    Color.red, Color.red, Color.red, Color.red, Color.red, Color.red,
    // Arms
    Color.green, Color.green, Color.green, Color.green,
    // Legs
    Color.blue, Color.blue, Color.blue, Color.blue
    };

    // The width for the skeleton lines
    private float lineWidth;

    // The material for the key point objects
    private Material keypointMat;
    /// <summary>
    /// Create a line between the key point specified by the start and end point indices
    /// </summary>
    /// <param name="pairIndex"></param>
    /// <param name="startIndex"></param>
    /// <param name="endIndex"></param>
    /// <param name="width"></param>
    /// <param name="color"></param>
    private void InitializeLine(int pairIndex, float width, Color color)
    {
        int startIndex = jointPairs[pairIndex].Item1;
        int endIndex = jointPairs[pairIndex].Item2;

        // Create new line GameObject
        string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
        lines[pairIndex] = new GameObject(name);

        // Add LineRenderer component
        LineRenderer lineRenderer = lines[pairIndex].AddComponent<LineRenderer>();
        // Make LineRenderer Shader Unlit
        lineRenderer.material = new Material(Shader.Find("Unlit/Color"));
        // Set the material color
        lineRenderer.material.color = color;

        // The line will consist of two points
        lineRenderer.positionCount = 2;

        // Set the width from the start point
        lineRenderer.startWidth = width;
        // Set the width from the end point
        lineRenderer.endWidth = width;
    }

    /// <summary>
    /// Initialize the pose skeleton
    /// </summary>
    private void InitializeSkeleton()
    {
        for (int i = 0; i < jointPairs.Length; i++)
        {
            InitializeLine(i, lineWidth, colors[i]);
        }
    }

    public PoseSkeleton(float pointScale = 0.1f, float lineWidth = 0.002f)
    {
        // Create one shared material for all the keypoints and lines
        sharedMaterial = new Material(Shader.Find("Unlit/Color"));
        sharedMaterial.color = Color.yellow;

        // Initialize keypoints array
        this.keypoints = new Transform[NUM_KEYPOINTS];

        // Initialize each keypoint
        for (int i = 0; i < NUM_KEYPOINTS; i++)
        {
            GameObject keypointObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Transform keypointTransform = keypointObject.transform;

            keypointTransform.position = Vector3.zero; // Start at the origin
            keypointTransform.localScale = new Vector3(pointScale, pointScale, pointScale);
            keypointObject.GetComponent<MeshRenderer>().material = sharedMaterial;
            keypointObject.name = partNames[i];

            this.keypoints[i] = keypointTransform;
            keypointObject.SetActive(false); // Start with keypoints deactivated
        }

        // Initialize the lines array
        this.lines = new GameObject[NUM_KEYPOINTS]; // Assuming one line per keypoint for simplicity

        // Initialize lines (without creating new GameObjects here)
        this.lineWidth = lineWidth;

        // The number of joint pairs
        int numPairs = keypoints.Length + 1;
        // Initialize the lines array
        lines = new GameObject[numPairs];

        // Initialize the pose skeleton
        InitializeSkeleton();
    }


    /// <summary>
    /// Toggles visibility for the skeleton
    /// </summary>
    /// <param name="show"></param>
    public void ToggleSkeleton(bool show)
    {
        for (int i = 0; i < jointPairs.Length; i++)
        {
            lines[i].SetActive(show);
            keypoints[jointPairs[i].Item1].gameObject.SetActive(show);
            keypoints[jointPairs[i].Item2].gameObject.SetActive(show);
        }
    }

    /// <summary>
    /// Clean up skeleton GameObjects
    /// </summary>
    public void Cleanup()
    {
        for (int i = 0; i < jointPairs.Length; i++)
        {
            GameObject.Destroy(lines[i]);
            GameObject.Destroy(keypoints[jointPairs[i].Item1].gameObject);
            GameObject.Destroy(keypoints[jointPairs[i].Item2].gameObject);
        }
    }

    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    /// <param name="keypoints"></param>
    /// <param name="sourceScale"></param>
    /// <param name="sourceTexture"></param>
    /// <param name="mirrorImage"></param>
    /// <param name="minConfidence"></param>
    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    /// <param name="keypoints">An array of keypoints with 2D positions and scores</param>
    /// <param name="cameraTransform">The transform of the main camera to position the keypoints in world space</param>
    /// <param name="minConfidence">Minimum confidence to show a keypoint</param>
    public void UpdateKeyPointPositions(Utils.Keypoint[] keypoints, Transform cameraTransform, float minConfidence)
    {
        for (int k = 0; k < keypoints.Length; k++)
        {
            if (keypoints[k].score >= minConfidence / 100f)
            {
                this.keypoints[k].gameObject.SetActive(true);

                // Translate the 2D screen point to a 3D point in world space
                Vector3 screenPoint = new Vector3(keypoints[k].position.x, keypoints[k].position.y, Camera.main.nearClipPlane + 1);
                Vector3 worldPoint = Camera.main.ScreenToWorldPoint(screenPoint);

                // Update the keypoint position
                this.keypoints[k].position = worldPoint;
            }
            else
            {
                this.keypoints[k].gameObject.SetActive(false);
            }
        }
        UpdateLines();
    }

    /// <summary>
    /// Draw the pose skeleton based on the latest location data
    /// </summary>
    public void UpdateLines()
    {
        for (int i = 0; i < jointPairs.Length; i++)
        {
            Transform startJoint = keypoints[jointPairs[i].Item1];
            Transform endJoint = keypoints[jointPairs[i].Item2];

            // Only proceed if both joints are active
            if (startJoint.gameObject.activeSelf && endJoint.gameObject.activeSelf)
            {
                // Ensure the line GameObject exists; create it if it doesn't
                if (lines[i] == null)
                {
                    lines[i] = new GameObject($"Line {i}");
                    LineRenderer lineRenderer = lines[i].AddComponent<LineRenderer>();
                    lineRenderer.material = sharedMaterial;
                    lineRenderer.startWidth = lineWidth;
                    lineRenderer.endWidth = lineWidth;
                    lineRenderer.positionCount = 2;
                }

                // Now, update the line positions
                LineRenderer lr = lines[i].GetComponent<LineRenderer>();
                lr.SetPosition(0, startJoint.position);
                lr.SetPosition(1, endJoint.position);
                lines[i].SetActive(true);
            }
            else
            {
                // Deactivate the line if it exists and either joint is inactive
                if (lines[i] != null)
                {
                    lines[i].SetActive(false);
                }
            }
        }
    }
}