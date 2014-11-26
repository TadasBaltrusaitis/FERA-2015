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
using PsycheInterop;
using System.Threading;
using Microsoft.Win32;

namespace Psyche
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class ChooseInput : Window
    {
        bool continueListing = true;
        AutoResetEvent doneListing = new AutoResetEvent(false);

        public ChooseInput()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            new Thread(EnumerateCameras).Start();
        }

        private void EnumerateCameras()
        {
            Thread.CurrentThread.IsBackground = true;
            int i = 0;
            while (continueListing)
            {
                using (Capture c = new Capture(i))
                {
                    try
                    {
                        var frame = c.GetNextFrame();

                        if (frame.Width == 0)
                            break;

                        var b = frame.CreateWriteableBitmap();
                        frame.UpdateWriteableBitmap(b);
                        b.Freeze();

                        Dispatcher.Invoke(() =>
                        {
                            int idx = i;
                            Image img = new Image();
                            img.Source = b;
                            img.Margin = new Thickness(20);
                            camerasPanel.ColumnDefinitions.Add(new ColumnDefinition());
                            img.SetValue(Grid.ColumnProperty, i);
                            img.SetValue(Grid.RowProperty, 1);
                            camerasPanel.Children.Add(img);

                            img.MouseDown += (s, e) =>
                            {
                                ChooseCamera(idx);
                            };
                        });
                    }
                    catch (PsycheInterop.CaptureFailedException)
                    {
                        break;
                    }

                }
                i++;
            }

            doneListing.Set();

        }

        private void OpenFile_Click(object sender, RoutedEventArgs e)
        {
            var d = new OpenFileDialog();
            if (d.ShowDialog(this) == true)
            {
                MainWindow window = new MainWindow(d.FileName);
                window.Show();
                Close();
            }
        }

        private void ChooseCamera(int idx)
        {
            new Thread(() => {
                continueListing = false;
                doneListing.WaitOne();
                Console.WriteLine("Select camera " + idx);

                Dispatcher.Invoke(() =>
                {
                    MainWindow window = new MainWindow(idx);
                    window.Show();
                    Close();
                });
            }).Start();
        }
    }
}
