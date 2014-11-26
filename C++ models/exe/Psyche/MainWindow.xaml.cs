using OpenCVWrappers;
using PsycheInterop;
using PsycheInterop.CLMTracker;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Psyche
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        #region High-Resolution Timing
        static DateTime startTime;
        static Stopwatch sw = new Stopwatch();
        static MainWindow()
        {
            startTime = DateTime.Now;
            sw.Start();
        }
        public static DateTime CurrentTime
        {
            get { return startTime + sw.Elapsed; }
        }
        #endregion


        private Capture capture;
        private WriteableBitmap latestImg;
        private WriteableBitmap latestAlignedFace;
        private WriteableBitmap latestHOGDescriptor;
        private bool reset = false;
        Point? resetPoint = null;

        BlockingCollection<Tuple<RawImage, RawImage>> frameQueue = new BlockingCollection<Tuple<RawImage, RawImage>>(2);

        FpsTracker videoFps = new FpsTracker();
        FpsTracker trackingFps = new FpsTracker();

        volatile bool detectionSucceeding = false;

        double fpsLimit = 0;
        string videoFile = null;

        Queue<Tuple<DateTime, string>> emotionLabelHistory = new Queue<Tuple<DateTime, string>>();

        public MainWindow(int device)
        {
            InitializeComponent();

            if (SystemParameters.PrimaryScreenWidth <= Width || SystemParameters.PrimaryScreenHeight <= Height)
                WindowState = System.Windows.WindowState.Maximized;

            capture = new Capture(device);

            new Thread(CaptureLoop).Start();
            new Thread(ProcessLoop).Start();
        }

        public MainWindow(string videoFile)
        {
            InitializeComponent();

            if (SystemParameters.PrimaryScreenWidth <= Width || SystemParameters.PrimaryScreenHeight <= Height)
                WindowState = System.Windows.WindowState.Maximized;

            this.videoFile = videoFile;
            capture = new Capture(videoFile);
            fpsLimit = capture.GetFPS();

            new Thread(CaptureLoop).Start();
            new Thread(ProcessLoop).Start();
        }

        private void CaptureLoop()
        {
            Thread.CurrentThread.IsBackground = true;

            var lastFrameTime = CurrentTime;

            while (true)
            {

                //////////////////////////////////////////////
                // CAPTURE FRAME AND DETECT LANDMARKS
                //////////////////////////////////////////////

                if (fpsLimit > 0)
                {
                    while (CurrentTime < lastFrameTime + TimeSpan.FromSeconds(1 / fpsLimit))
                        Thread.Sleep(1);
                }


                RawImage frame = null;
                try
                {
                    frame = capture.GetNextFrame();
                }
                catch (PsycheInterop.CaptureFailedException)
                {
                    if (videoFile != null)
                    {
                        capture = new Capture(videoFile);
                        fpsLimit = capture.GetFPS();

                        new Thread(CaptureLoop).Start();
                    }
                    break;
                }
                lastFrameTime = CurrentTime;
                videoFps.AddFrame();

                var grayFrame = capture.GetCurrentFrameGray();

                if (grayFrame == null)
                    continue;

                frameQueue.TryAdd(new Tuple<RawImage, RawImage>(frame, grayFrame));

                try
                {
                    Dispatcher.Invoke(() =>
                    {
                        if (latestImg == null)
                            latestImg = frame.CreateWriteableBitmap();

                        fpsLabel.Content = "Video: " + videoFps.GetFPS().ToString("0") + " FPS | Tracking: " + trackingFps.GetFPS().ToString("0") + " FPS";

                        if (!detectionSucceeding)
                        {
                            frame.UpdateWriteableBitmap(latestImg);

                            video.OverlayLines.Clear();
                            video.OverlayPoints.Clear();

                            video.Source = latestImg;
                        }
                    });
                }
                catch (TaskCanceledException)
                {
                    // Quitting
                    break;
                }
            }
        }

        private void ProcessLoop()
        {
            Thread.CurrentThread.IsBackground = true;

            CLMParameters clmParams = new CLMParameters();
            CLM clmModel = new CLM();
            float fx = 500, fy = 500, cx = 0, cy = 0;

            FaceAnalyser analyser = new FaceAnalyser();

            DateTime? startTime = CurrentTime;

            arousalPlot.AssocColor(0, Colors.Red);
            valencePlot.AssocColor(0, Colors.Blue);

            while (true)
            {
                var newFrames = frameQueue.Take();

                var frame = new RawImage(newFrames.Item1);
                var grayFrame = newFrames.Item2;

                if (!startTime.HasValue)
                    startTime = CurrentTime;

                if (cx == 0 && cy == 0)
                {
                    cx = grayFrame.Width / 2f;
                    cy = grayFrame.Height / 2f;
                }

                if (reset)
                {
                    clmModel.Reset();
                    analyser.Reset();
                    reset = false;
                }

                if (resetPoint.HasValue)
                {
                    clmModel.Reset(resetPoint.Value.X, resetPoint.Value.Y);
                    analyser.Reset();
                    resetPoint = null;
                }

                detectionSucceeding = clmModel.DetectLandmarksInVideo(grayFrame, clmParams);

                List<Tuple<Point, Point>> lines = null;
                List<Point> landmarks = null;
                if (detectionSucceeding)
                {
                    landmarks = clmModel.CalculateLandmarks();
                    lines = clmModel.CalculateBox(fx, fy, cx, cy);
                }
                else
                {
                    analyser.Reset();
                }

                //////////////////////////////////////////////
                // Analyse frame and detect AUs
                //////////////////////////////////////////////

                analyser.AddNextFrame(grayFrame, clmModel, (CurrentTime - startTime.Value).TotalSeconds);

                var alignedFace = analyser.GetLatestAlignedFace();
                var hogDescriptor = analyser.GetLatestHOGDescriptorVisualisation();

                trackingFps.AddFrame();

                Dictionary<String, double> aus = analyser.GetCurrentAUs();
                string emotion = analyser.GetCurrentCategoricalEmotion();
                double arousal = analyser.GetCurrentArousal();
                double valence = analyser.GetCurrentValence();
                double confidence = analyser.GetConfidence();
                try
                {
                    Dispatcher.Invoke(() =>
                    {

                        if (latestAlignedFace == null)
                            latestAlignedFace = alignedFace.CreateWriteableBitmap();

                        if (latestHOGDescriptor == null)
                            latestHOGDescriptor = hogDescriptor.CreateWriteableBitmap();

                        confidenceBar.Value = confidence;

                        if (detectionSucceeding)
                        {

                            frame.UpdateWriteableBitmap(latestImg);
                            alignedFace.UpdateWriteableBitmap(latestAlignedFace);
                            hogDescriptor.UpdateWriteableBitmap(latestHOGDescriptor);

                            imgAlignedFace.Source = latestAlignedFace;
                            imgHOGDescriptor.Source = latestHOGDescriptor;

                            video.OverlayLines = lines;
                            video.OverlayPoints = landmarks;
                            video.Confidence = confidence;

                            video.Source = latestImg;

                            Dictionary<int, double> arousalDict = new Dictionary<int, double>();
                            arousalDict[0] = arousal * 0.5 + 0.5;
                            arousalPlot.AddDataPoint(new DataPoint() { Time = CurrentTime, values = arousalDict, Confidence = confidence });

                            Dictionary<int, double> valenceDict = new Dictionary<int, double>();
                            valenceDict[0] = valence * 0.5 + 0.5;
                            valencePlot.AddDataPoint(new DataPoint() { Time = CurrentTime, values = valenceDict, Confidence = confidence });

                            Dictionary<int, double> avDict = new Dictionary<int, double>();
                            avDict[0] = arousal;
                            avDict[1] = valence;
                            avPlot.AddDataPoint(new DataPoint() { Time = CurrentTime, values = avDict, Confidence = confidence });

                            auGraph.Update(aus, confidence);

                            emotionLabelHistory.Enqueue(new Tuple<DateTime, string>(CurrentTime, emotion));

                            UpdateEmotionLabel();
                        }
                        else
                        {
                            foreach (var k in aus.Keys.ToArray())
                                aus[k] = 0;

                            auGraph.Update(aus, 0);
                        }
                    });
                }
                catch (TaskCanceledException)
                {
                    // Quitting
                    break;
                }
            }
        }

        private void QuitButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            reset = true;
        }

        private void video_MouseDown(object sender, MouseButtonEventArgs e)
        {
            var clickPos = e.GetPosition(video);

            resetPoint = new Point(clickPos.X / video.ActualWidth, clickPos.Y / video.ActualHeight);
        }

        private void UpdateEmotionLabel()
        {

            while (emotionLabelHistory.Peek().Item1 < CurrentTime - TimeSpan.FromSeconds(0.8))
                emotionLabelHistory.Dequeue();

            string emotion = emotionLabelHistory.Peek().Item2;
            foreach (var t in emotionLabelHistory)
            {
                if (t.Item2 != emotion)
                    return;
            }
            emotionLabel.Content = emotion;
        }

    }
}
