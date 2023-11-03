using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
using Unity.Barracuda;
using System.Threading;
using System.Diagnostics;
using System;
using System.Reflection;
using System.Linq;
using UnityEngine.UI;
using TMPro;
#if ENABLE_WINMD_SUPPORT
using Windows.Storage;
using Windows.Storage.Streams;
using System.IO;
#endif

public class PoseEstimator : MonoBehaviour
{
    public enum ModelType
    {
        MobileNet,
        ResNet50
    }

    public enum EstimationType
    {
        MultiPose,
        SinglePose
    }

    [Tooltip("The maximum number of posees to estimate")]
    [Range(1, 20)]
    public int maxPoses = 1;

    [Tooltip("The size of the pose skeleton key points")]
    public float pointScale = 10f;

    [Tooltip("The width of the pose skeleton lines")]
    public float lineWidth = 5f;

    [Tooltip("The minimum confidence level required to display the key point")]
    [Range(0, 100)]
    public int minConfidence = 30;

    [Tooltip("The type of pose estimation to be performed")]
    public EstimationType estimationType = EstimationType.SinglePose;

    [Tooltip("The MobileNet model asset file to use when performing inference")]
    public NNModel mobileNetModelAsset;

    [Tooltip("The ResNet50 model asset file to use when performing inference")]
    public NNModel resnetModelAsset;

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The ComputeShader that will perform the model-specific preprocessing")]
    public ComputeShader posenetShader;

    [Tooltip("The model architecture used")]
    public ModelType modelType = ModelType.ResNet50;

   
    [Tooltip("The dimensions of the image being fed to the model")]
    public Vector2Int imageDims = new Vector2Int(512, 512);


    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;


    public MeshRenderer imageRenderer;

    // Assuming you've assigned this in the Inspector
    public TextMeshProUGUI textTime;




    // Inference time recording
    List<double> results = new List<double>();
    List<double> resultsIter = new List<double>();
    List<double> resultsLoad = new List<double>();
    List<double> resultsPreProcessImg = new List<double>();
    List<double> resultsDispose = new List<double>();
    List<double> resultsPostProcess = new List<double>();
    List<double> resultsPostProcessPeekOutput = new List<double>();
    List<double> resultsDecodeSinglePose = new List<double>();
    List<double> resultsUpdateSkeleton = new List<double>();
    

    // initialise time recorders
    Stopwatch swInf = new Stopwatch(); //only this one i store in a list and only after i calculate mean and std
    Stopwatch swIter = new Stopwatch(); //times of each loop
    Stopwatch swExec = new Stopwatch();
    Stopwatch swLoadImg = new Stopwatch();
    Stopwatch swPreProcessImg = new Stopwatch(); //preprocessing + inference
    Stopwatch swPostProcessImg = new Stopwatch(); //preprocessing + inference
    Stopwatch swPostProcessImgPeekOutput = new Stopwatch();
    Stopwatch swDispose = new Stopwatch();
    Stopwatch swDecodeSinglePose = new Stopwatch();
    Stopwatch swSkeleton = new Stopwatch();

    // Stores the current estimated 2D keypoint locations in videoTexture
    private Utils.Keypoint[][] poses;

    // Live video input from a webcam
    private WebCamTexture webcamTexture;

    // The dimensions of the current video source (dimensions of either video or webcam or image)
    private Vector2Int videoDims;
 
    // Target dimensions for model input
    private Vector2Int targetDims;

    // Used to scale the input image dimensions while maintaining aspect ratio
    private float aspectRatioScale;

    // The texture used to create input tensor
    private RenderTexture rTex;

    // The preprocessing function for the current model type
    private System.Action<float[]> preProcessFunction;

    // Stores the input data for the model
    private Tensor input;

    // Array of pose skeletons
    private PoseSkeleton[] skeletons;

    /// <summary>
    /// Keeps track of the current inference backend, model execution interface, 
    /// and model type
    /// </summary>
    private struct Engine
    {
        public WorkerFactory.Type workerType;
        public IWorker worker;
        public ModelType modelType;

        public Engine(WorkerFactory.Type workerType, Model model, ModelType modelType)
        {
            this.workerType = workerType;
            worker = WorkerFactory.CreateWorker(workerType, model);
            this.modelType = modelType;
        }
    }

    // The interface used to execute the neural network
    private Engine engine;

    // The name for the heatmap layer in the model asset
    private string heatmapLayer;

    // The name for the offsets layer in the model asset
    private string offsetsLayer;

    // The name for the forwards displacement layer in the model asset
    private string displacementFWDLayer;

    // The name for the backwards displacement layer in the model asset
    private string displacementBWDLayer;

    // The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "heatmap_predictions";



    /// <summary>
    /// Prepares the videoScreen GameObject to display the chosen video source.
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <summary>
    /// Updates the output layer names based on the selected model architecture
    /// and initializes the Barracuda inference engine witht the selected model.
    /// </summary>
    /// 

    /// <summary>
    /// Initialize pose skeletons
    /// </summary>
    private void InitializeSkeletons()
    {
        // Initialize the list of pose skeletons
        if (estimationType == EstimationType.SinglePose) maxPoses = 1;
        skeletons = new PoseSkeleton[maxPoses];

        // Populate the list of pose skeletons
        for (int i = 0; i < maxPoses; i++) skeletons[i] = new PoseSkeleton(pointScale, lineWidth);
    }


    private void InitializeBarracuda()
    {
        // The compiled model used for performing inference
        Model m_RunTimeModel;

        if (modelType == ModelType.MobileNet)
        {
            preProcessFunction = Utils.PreprocessMobileNet;
            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(mobileNetModelAsset);
            displacementFWDLayer = m_RunTimeModel.outputs[2];
            displacementBWDLayer = m_RunTimeModel.outputs[3];
        }
        else
        {
            preProcessFunction = Utils.PreprocessResNet;
            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(resnetModelAsset);
            displacementFWDLayer = m_RunTimeModel.outputs[3];
            displacementBWDLayer = m_RunTimeModel.outputs[2];
        }

        heatmapLayer = m_RunTimeModel.outputs[0];
        offsetsLayer = m_RunTimeModel.outputs[1];

        // Create a model builder to modify the m_RunTimeModel
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);

        // Add a new Sigmoid layer that takes the output of the heatmap layer
        modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

        // Validate if backend is supported, otherwise use fallback type.
        workerType = WorkerFactory.ValidateType(workerType);

        // Create a worker that will execute the model with the selected backend
        engine = new Engine(workerType, modelBuilder.model, modelType);
    }


   
    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns></returns>
    private void ProcessImageGPU(RenderTexture image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }


      

    /// <summary>
    /// Obtains the model output and either decodes single or mutlple poses
    /// </summary>
    /// <param name="engine"></param>
    private void ProcessOutput(IWorker engine)
    {
        swPostProcessImg.Reset();
        swPostProcessImg.Start();

        swPostProcessImgPeekOutput.Reset();
        swPostProcessImgPeekOutput.Start();
        // Get the model output
        Tensor heatmaps = engine.PeekOutput(predictionLayer);
        Tensor offsets = engine.PeekOutput(offsetsLayer);
        Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
        Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);
        swPostProcessImgPeekOutput.Stop();
        resultsPostProcessPeekOutput.Add(swPostProcessImgPeekOutput.ElapsedMilliseconds);

        // Calculate the stride used to scale down the inputImage
        int stride = (imageDims.y - 1) / (heatmaps.shape.height - 1);
        stride -= (stride % 8);

        if (estimationType == EstimationType.SinglePose)
        {
            // Initialize the array of Keypoint arrays
            poses = new Utils.Keypoint[1][];

            // Determine the key point locations
            swDecodeSinglePose.Reset();
            swDecodeSinglePose.Start();
            poses[0] = Utils.DecodeSinglePose(heatmaps, offsets, stride);
            swDecodeSinglePose.Stop();
            resultsDecodeSinglePose.Add(swDecodeSinglePose.ElapsedMilliseconds);
        }
        else
        {
            //https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-6/
        }

        // Release the resources allocated for the output Tensors
        heatmaps.Dispose();
        offsets.Dispose();
        displacementFWD.Dispose();
        displacementBWD.Dispose();

        swPostProcessImg.Stop();
        resultsPostProcess.Add(swPostProcessImg.ElapsedMilliseconds);
    }

    public string GetAllPublicVariablesAsString()
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();

        // Get all public instance variables of this object
        var fields = this.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance);

        foreach (var field in fields)
        {
            // Get the value of the field
            var value = field.GetValue(this);

            // Append the field name and value to the string builder
            sb.AppendLine($"{field.Name}: {value}");
        }

        return sb.ToString();
    }
    //void displayImage(RenderTexture tmpRTex)
    //{
    //    // Assume rTex is your RenderTexture
    //    debugImageDisplay.texture = tmpRTex;
    //}
    public string GetModelName()
    {
        var modelAssetField = this.GetType().GetField("resnetModelAsset", BindingFlags.Public | BindingFlags.Instance);
        if (modelAssetField != null)
        {
            var modelAsset = modelAssetField.GetValue(this);
            if (modelAsset != null)
            {
                string modelAssetString = modelAsset.ToString();
                var modelNameWithBrackets = modelAssetString.Split(' ')[0];
                var modelName = modelNameWithBrackets.Trim('(').Trim(')');
                return modelName;
            }
        }

        return null;
    }

#if ENABLE_WINMD_SUPPORT
    // Writing a file in UWP environment
    void WriteFileUWP(string content)
    {
        UnityEngine.Debug.Log("Saving result WriteFileUWP...");
        string modelName = GetModelName();
        string fileName = $"{modelName}_results_{System.DateTime.Now.Ticks}.txt";
        using (var fileStream = new FileStream(fileName, FileMode.Append))
        {
            using (var streamWriter = new StreamWriter(fileStream))
            {
                UnityEngine.Debug.Log("Saving result WriteFileUWP WriteLine...");
                streamWriter.WriteLine(content);
                UnityEngine.Debug.Log("Saving result WriteFileUWP WriteLine DONE...");
            }
        }

        string sourcePath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
        string destinationPath = Path.Combine(Windows.Storage.ApplicationData.Current.LocalFolder.Path, fileName);

        System.IO.File.Copy(sourcePath, destinationPath, true);
        UnityEngine.Debug.Log($"Copied file to {destinationPath}");
    }


#else
    // Writing a file in Unity Editor
    void WriteFileUnity(string content)
    {
        // create a new file, or overwrite an existing one
        // TO DO: add datetime
        UnityEngine.Debug.Log("Saving result WriteFileUnity...");
        string modelName = GetModelName();
        System.IO.File.WriteAllText($"{modelName}_results_test_.txt", content);
    }
#endif

    /// <summary>
    /// Store inference times restults
    /// </summary>
    /// <param name="content"></param>
    void SaveResult(string content)
    {
        UnityEngine.Debug.Log("Saving result...");
#if ENABLE_WINMD_SUPPORT
        // Writing a file in UWP environment
        WriteFileUWP(content);
#else
        // Writing a file in for local testing in Unity Editor
        WriteFileUnity(content);
#endif

    }

    /// <summary>
    /// Execute neural network on single image
    /// </summary>
    /// <param name="tex"></param>
    

    private void Inference(Boolean storeTime = true)
    {
        //inference times
        //inference times
        swInf.Reset(); // Reset the stopwatch before starting it again
        swInf.Start();
        // Execute neural network with the provided input
        engine.worker.Execute(input);
        engine.worker.FlushSchedule();
        swInf.Stop();

        if (storeTime) { results.Add(swInf.ElapsedMilliseconds); }

        // Release GPU resources allocated for the Tensor
        swDispose.Reset();
        swDispose.Start();

        //in cpu is not needed
        if (workerType == WorkerFactory.Type.ComputePrecompiled) input.Dispose();
        swDispose.Stop();
        resultsDispose.Add(swDispose.ElapsedMilliseconds);

        //display inference time

        textTime.text = $"{swInf.ElapsedMilliseconds}";
    }

    double CalculateMean(List<double> data)
    {
        if (data.Count > 1)
        {
            return data.Skip(1).Average(); //skip first image 
        }
        else
        {
            // handle the case where there is 0 or 1 element in the list
            return double.NaN;
        }
    }

    double CalculateStandardDeviation(List<double> data, double mean)
    {
        if (data.Count > 1)
        {
            double sumOfSquaresOfDifferences = data.Skip(1).Select(val => (val - mean) * (val - mean)).Sum(); //skip first image 
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (data.Count - 1));
            return standardDeviation;
        }
        else
        {
            // handle the case where there is 0 or 1 element in the list
            return double.NaN;
        }
    }


    // Start is called before the first frame update
    void Start()
    {
        
        // Limit application framerate to the target webcam framerate
        Application.targetFrameRate = webcamFPS;

        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);

        // Start the Camera
        webcamTexture.Play();

        // Initialize the RenderTexture that will store the processed input image
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);


        // Initialize the Barracuda inference engine based on the selected model
        InitializeBarracuda();

        // Initialize pose skeletons
        InitializeSkeletons();
        swExec.Start();

    }

    

    void createAndSaveResults()
    {
        swExec.Stop();
        long deltaExec = swExec.ElapsedMilliseconds;
        string settings = GetAllPublicVariablesAsString();
        double meanInf = CalculateMean(results);
        double stdInf = CalculateStandardDeviation(results, meanInf);
        double meanIter = CalculateMean(resultsIter);
        double stdIter = CalculateStandardDeviation(resultsIter, meanIter);
        double meanLoadImg = CalculateMean(resultsLoad);
        double stdLoadImg = CalculateStandardDeviation(resultsLoad, meanLoadImg);
        double meanPreProcessImg = CalculateMean(resultsPreProcessImg);
        double stdPreProcessImg = CalculateStandardDeviation(resultsPreProcessImg, meanPreProcessImg);
        double meanDispose = CalculateMean(resultsDispose);
        double stdDispose = CalculateStandardDeviation(resultsDispose, meanDispose);
        double meanPostProcessImg = CalculateMean(resultsPostProcess);
        double stdPostProcessImg = CalculateStandardDeviation(resultsPostProcess, meanPostProcessImg);
        double meanPostProcessImgPeekOutput = CalculateMean(resultsPostProcessPeekOutput);
        double stdPostProcessImgPeekOutput = CalculateStandardDeviation(resultsPostProcessPeekOutput, meanPostProcessImgPeekOutput);
        double meanDecodeSinglePose = CalculateMean(resultsDecodeSinglePose);
        double stdDecodeSinglePose = CalculateStandardDeviation(resultsDecodeSinglePose, meanDecodeSinglePose);
        double meanSkeleton = CalculateMean(resultsUpdateSkeleton);
        double stdSkeleton = CalculateStandardDeviation(resultsUpdateSkeleton, meanSkeleton);



        SaveResult("Inference times:\n\n" + String.Join("\n", results) +
            "\n\nIteration times:\n" + String.Join("\n", resultsIter) +
            "\n\n Experiment Info:\n" + settings +
            "\nSTATISTICS:\n" +
            $"\ndeltaExec: { deltaExec}" +
            $"\nmeanIter: {meanIter}\nstdIter: {stdIter}" +
            $"\nmeanLoadImg: {meanLoadImg}\nstdLoadImg: {stdLoadImg}" +
            $"\nmeanPreProcessImg: {meanPreProcessImg}\nstdPreProcesssImg: {stdPreProcessImg}" +
            $"\nmeanInf: {meanInf}\nStdInf: {stdInf}" +
            $"\nmeanDispose: {meanDispose}\nStdDispose: {stdDispose}" +
            $"\nmeanPostProcessImg: {meanPostProcessImg}\nstdPostProcessImg: {stdPostProcessImg}" +
            $"\nmeanPostProcessImgPeekOutput: {meanPostProcessImgPeekOutput}\nstdPostProcessImgPeekOutput: {stdPostProcessImgPeekOutput}" +
            $"\nmeanDecodeSinglePose: {meanDecodeSinglePose}\nstdPostProcessDecodeSinglePose: {stdDecodeSinglePose}" +
            $"\nmeanUpdateSkeleton: {meanSkeleton}\nstdPreProcessImg: {stdSkeleton}"
           );
    }
    // Update is called once per frame
    void Update()
    {

        swIter.Reset();
        swIter.Start();

        if (webcamTexture.didUpdateThisFrame)
        {
            // Determine aspect ratio of the source texture
            float sourceAspectRatio = (float)webcamTexture.width / webcamTexture.height;

            // Calculate the target dimensions while maintaining the aspect ratio
            Vector2Int targetDims = CalculateTargetDimensions(webcamTexture.width, webcamTexture.height, imageDims);

            // Ensure we have a render texture with the correct user-specified dimensions
            if (rTex == null || rTex.width != targetDims.x || rTex.height != targetDims.y)
            {
                // Release the old RenderTexture if it was created previously
                if (rTex != null)
                    RenderTexture.ReleaseTemporary(rTex);

                // Create a new RenderTexture with the target dimensions
                rTex = RenderTexture.GetTemporary(targetDims.x, targetDims.y, 24, RenderTextureFormat.ARGB32);
            }

            // Now rTex has the correct dimensions as specified by the user
            // Copy the webcamTexture to rTex, potentially using a material for resizing if needed
            Graphics.Blit(webcamTexture, rTex); // Add a material with a resizing shader if aspect ratio correction is needed

            // Apply preprocessing steps
            ProcessImageGPU(rTex, preProcessFunction.Method.Name);

            // Create a Tensor of shape [1, image.height, image.width, 3]
            input = new Tensor(rTex, channels: 3);


            // Reinitialize Barracuda with the selected model and backend 
            if (engine.modelType != modelType || engine.workerType != workerType)
            {
                engine.worker.Dispose();
                InitializeBarracuda();
            }

            // Execute neural network with the provided input
            Inference(false);
            // Release GPU resources allocated for the Tensor
            //input.Dispose();
            // Decode the keypoint coordinates from the model output
            ProcessOutput(engine.worker);
            swSkeleton.Reset();
            swSkeleton.Start();

            //updateSkeleton();

            swSkeleton.Stop();
            resultsUpdateSkeleton.Add(swSkeleton.ElapsedMilliseconds);

            swIter.Stop();
            resultsIter.Add(swIter.ElapsedMilliseconds);


        }

    }

    Vector2Int CalculateTargetDimensions(int sourceWidth, int sourceHeight, Vector2Int targetSize)
    {
        // Calculate target aspect ratio based on user-specified dimensions
        float targetAspectRatio = (float)targetSize.x / targetSize.y;

        // Calculate new dimensions
        int newWidth, newHeight;
        if (targetAspectRatio > (float)sourceWidth / sourceHeight)
        {
            // If target aspect ratio is greater than source, match height and scale width
            newHeight = targetSize.y;
            newWidth = Mathf.RoundToInt(sourceWidth * (targetSize.y / (float)sourceHeight));
        }
        else
        {
            // If target aspect ratio is less or equal, match width and scale height
            newWidth = targetSize.x;
            newHeight = Mathf.RoundToInt(sourceHeight * (targetSize.x / (float)sourceWidth));
        }

        // Ensure dimensions are not smaller than the minimum expected by the model
        newWidth = Mathf.Max(newWidth, 130); // Example minimum width
        newHeight = Mathf.Max(newHeight, 130); // Example minimum height

        return new Vector2Int(newWidth, newHeight);
    }


    //void updateSkeleton()
    //{
    //    // Reinitialize pose skeletons
    //    if (maxPoses != skeletons.Length)
    //    {
    //        foreach (PoseSkeleton skeleton in skeletons)
    //        {
    //            skeleton.Cleanup();
    //        }

    //        // Initialize pose skeletons
    //        InitializeSkeletons();
    //    }

    //    // The smallest dimension of the videoTexture
    //    int minDimension = Mathf.Min(webcamTexture.width, webcamTexture.height);

    //    // The value used to scale the key point locations up to the source resolution
    //    float scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

    //    // Update the pose skeletons
    //    for (int i = 0; i < skeletons.Length; i++)
    //    {
    //        if (i <= poses.Length - 1)
    //        {
    //            skeletons[i].ToggleSkeleton(true);

    //            // Update the positions for the key point GameObjects
    //            skeletons[i].UpdateKeyPointPositions(poses[i], scale, videoTexture, mirrorScreen, minConfidence);
    //            skeletons[i].UpdateLines();
    //        }
    //        else
    //        {
    //            skeletons[i].ToggleSkeleton(false);
    //        }
    //    }

    //}


    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.worker.Dispose();
    }


}