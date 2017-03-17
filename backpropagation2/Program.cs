using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace backpropagation2
{
    class Program
    {
        static void Main(string[] args)
        {
            /*1. final architecture
            2.parameter setting
            3. flowchart
            4. analysys of result
             -% accuracy
             -consusion matrix
             */
            try
            {
                int epoch = 500;
                int Znode = 50;
                int Ynode = 1;
                double learningRate = 0.5;
                double numberOfClasses = 3.0;

                bacpropagationProcessing bp = new bacpropagationProcessing();
                // First run training then comment the training and uncomment the testing
            bp.training(epoch,"c:/users/firdaus samad/documents/visual studio 2015/Projects/backpropagation2/backpropagation2/trainingInput.txt", Znode, Ynode, learningRate,numberOfClasses);

         //   bp.testing("c:/users/firdaus samad/documents/visual studio 2015/Projects/backpropagation2/backpropagation2/testingInput.txt", Znode, Ynode, learningRate,numberOfClasses);

            }

            catch (ArgumentException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (OutOfMemoryException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (DirectoryNotFoundException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (IOException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            Console.ReadLine();
        }
    }
}
