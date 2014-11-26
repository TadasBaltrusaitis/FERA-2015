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

namespace Psyche
{
    /// <summary>
    /// Interaction logic for MultiBarGraph.xaml
    /// </summary>
    public partial class MultiBarGraph : UserControl
    {
        Dictionary<string, BarGraph> bars = new Dictionary<string, BarGraph>();

        public MultiBarGraph()
        {
            InitializeComponent();
        }

        public void Update(Dictionary<string, double> data, double confidence)
        {
            foreach (var kvp in data) 
            {
                var title = kvp.Key;
                var value = kvp.Value;

                if (!bars.ContainsKey(title))
                {
                    BarGraph newBar = new BarGraph();
                    newBar.Title = title.Replace(" ", "\n");
                    barGrid.ColumnDefinitions.Add(new ColumnDefinition());
                    Grid.SetColumn(newBar, bars.Count);
                    barGrid.Children.Add(newBar);
                    bars[title] = newBar;
                }

                var bar = bars[title];
                bar.SetConfidence(confidence);
                bar.SetValue(value / 5);
            }
        }

    }
}
