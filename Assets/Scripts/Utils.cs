using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;


public class Utils
{
    /// <summary>
    /// Stores the heatmap score, position, and partName index for a single keypoint
    /// </summary>
    public struct Keypoint
    {
        public float score;
        public Vector2 position;
        public int id;

        public Keypoint(float score, Vector2 position, int id)
        {
            this.score = score;
            this.position = position;
            this.id = id;
        }
    }

    /// <summary>
    /// Get the offset values for the provided heatmap indices
    /// </summary>
    /// <param name="y">Heatmap column index</param>
    /// <param name="x">Heatmap row index</param>
    /// <param name="keypoint">Heatmap channel index</param>
    /// <param name="offsets">Offsets output tensor</param>
    /// <returns></returns>
    public static Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
    {
        // Get the offset values for the provided heatmap coordinates
        return new Vector2(offsets[0, y, x, keypoint + 17], offsets[0, y, x, keypoint]);
    }
    /// <summary>
    /// Applies the preprocessing steps for the MobileNet model on the CPU
    /// </summary>
    /// <param name="tensor">Pixel data from the input tensor</param>

    /// <summary>
    /// Calculate the position of the provided key point in the input image
    /// </summary>
    /// <param name="part"></param>
    /// <param name="stride"></param>
    /// <param name="offsets"></param>
    /// <returns></returns>
    public static Vector2 GetImageCoords(Keypoint part, int stride, Tensor offsets)
    {
        // The accompanying offset vector for the current coords
        Vector2 offsetVector = GetOffsetVector((int)part.position.y, (int)part.position.x,
                                               part.id, offsets);

        // Scale the coordinates up to the input image resolution
        // Add the offset vectors to refine the key point location
        return (part.position * stride) + offsetVector;
    }

    /// <summary>
    /// Determine the estimated key point locations using the heatmaps and offsets tensors
    /// </summary>
    /// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
    /// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
    /// <returns>An array of keypoints for a single pose</returns>
    public static Keypoint[] DecodeSinglePose(Tensor heatmaps, Tensor offsets, int stride)
    {
        Keypoint[] keypoints = new Keypoint[heatmaps.channels];

        // Iterate through heatmaps
        for (int c = 0; c < heatmaps.channels; c++)
        {
            Keypoint part = new Keypoint();
            part.id = c;

            // Iterate through heatmap columns
            for (int y = 0; y < heatmaps.height; y++)
            {
                // Iterate through column rows
                for (int x = 0; x < heatmaps.width; x++)
                {
                    if (heatmaps[0, y, x, c] > part.score)
                    {
                        // Update the highest confidence for the current key point
                        part.score = heatmaps[0, y, x, c];

                        // Update the estimated key point coordinates
                        part.position.x = x;
                        part.position.y = y;
                    }
                }
            }

            // Calcluate the position in the input image for the current (x, y) coordinates
            part.position = GetImageCoords(part, stride, offsets);

            // Add the current keypoint to the list
            keypoints[c] = part;
        }

        return keypoints;
    }
    public static void PreprocessMobileNet(float[] tensor)
    {
        // Normaliz the values to the range [-1, 1]
        System.Threading.Tasks.Parallel.For(0, tensor.Length, (int i) =>
        {
            tensor[i] = (float)(2.0f * tensor[i] / 1.0f) - 1.0f;
        });
    }

    ///// <summary>
    ///// Applies the preprocessing steps for the ResNet50 model on the CPU
    ///// </summary>
    ///// <param name="tensor">Pixel data from the input tensor</param>
    public static void PreprocessResNet(float[] tensor)
    {
        System.Threading.Tasks.Parallel.For(0, tensor.Length / 3, (int i) =>
        {
            tensor[i * 3 + 0] = (float)tensor[i * 3 + 0] * 255f - 123.15f;
            tensor[i * 3 + 1] = (float)tensor[i * 3 + 1] * 255f - 115.90f;
            tensor[i * 3 + 2] = (float)tensor[i * 3 + 2] * 255f - 103.06f;
        });
    }

        // Start is called before the first frame update
        void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
