using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace Psyche
{
    /// <summary>
    /// Interaction logic for BarGraph.xaml
    /// </summary>
    public partial class BarGraph : UserControl
    {
        private double targetValue = 0;
        private double targetConfidence = 0;
        private double confidence = 0.01;

        public BarGraph()
        {
            InitializeComponent();
            DispatcherTimer dt = new DispatcherTimer(TimeSpan.FromMilliseconds(20), DispatcherPriority.Background, Timer_Tick, Dispatcher);

        }

        public void SetValue(double value)
        {
            targetValue = value;
        }

        public void SetConfidence(double confidence)
        {
            targetConfidence = confidence;
        }

        public string Title
        {
            get { return (string)lblTitle.Content; }
            set { lblTitle.Content = value; }
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            var minHeight = barContainer.ActualHeight * 0.1;
            bar.Height = (bar.Height-minHeight) * 0.9 + ((barContainer.ActualHeight-minHeight) * targetValue) * 0.1 + minHeight;

            if (confidence != targetConfidence)
            {
                confidence = confidence * 0.9 + targetConfidence * 0.1;

                if (Math.Abs(confidence - targetConfidence) < 0.0001)
                    confidence = targetConfidence;

                Color bTransparent = Colors.CadetBlue;
                bTransparent.A = 0;

                GradientStopCollection gs = new GradientStopCollection();
                gs.Add(new GradientStop(bTransparent, 0));
                gs.Add(new GradientStop(Colors.CadetBlue, 1));
                LinearGradientBrush g = new LinearGradientBrush(gs, new Point(0, 0), new Point(0, bar.Height*(1-confidence)));
                g.MappingMode = BrushMappingMode.Absolute;
                g.Freeze();
                bar.Fill = g;
            }
        }
    }
}
