using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace backpropagation2
{
    public class bacpropagationProcessing
    {
        public static int correctnessIrisSetosa = 0;
        public static int correctnessIrisVersicolour = 0;
        public static int correctnessIrisVirginica = 0;
        public static int incorrectnessIrisSetosa = 0;
        public static int incorrectnessIrisVersicolour = 0;
        public static int incorrectnessIrisVirginica = 0;

        private struct bpnn
        {
            public int num;
            public int inputNode;
            public int totalData;
            public double[] target;
            public int Inode;
            public double[,] data;
            public double[] x;
            public double[] deltaKError;
            public double[] Z_in;
            public double[] Z;
            public double[] Y_in;
            public double[] Y;
            public double[,] Vweight;
            public double[,] Wweight;
            public double[,] weightChangeV;
            public double[,] weightChangeW;
            public double sum2;
            public double meanSquareError;
        }

        // bpnn Training
        public void training(int epoch, string path, int Znode, int Ynode, double learningRate, double numberOfClasses)
        {
            Console.WriteLine("Training Result :");
            bpnn bp = new bpnn();
            bp.num = 1;
            bp.inputNode = 0;
            bp.totalData = 0;

            //read data from txt file
            using (StreamReader myReader = new StreamReader(path))
            {
                string line = " ";
                string l = "";
                while (line != null)
                {
                    line = myReader.ReadLine();
                    if (line != null)
                    {
                        l = line;
                        //Split data by comma
                        String[] value = l.Split(null);

                        //to get the number of input node
                        bp.inputNode = value.Length;
                        //to get the number of total iteration/ line of data(150 data)
                        bp.totalData = bp.totalData + 1;
                    }
                }
                myReader.Close();
            }
            //Read Input data
            bp.Inode = bp.inputNode - 1;
            bp.data = new double[bp.totalData, bp.Inode];
            bp.target = new double[bp.totalData];
            using (StreamReader myReader = new StreamReader(path))
            {
                string l = "";
                int a = 0;
                string line = myReader.ReadLine();
                while (line != null)
                {
                    l = line;
                    //Split data by space
                    String[] value = l.Split(null);

                    for (int i = 0; i < bp.inputNode; i++)
                    {
                        if (i == bp.inputNode - 1)
                            bp.target[a] = Convert.ToDouble(value[i]);
                        else {
                            bp.data[a, i] = Convert.ToDouble(value[i]);
                            // Console.WriteLine(bp.data[a, i]);
                        }
                    }
                    a++;

                    line = myReader.ReadLine();
                }
                myReader.Close();
            }
            bp.Vweight = new double[bp.Inode, Znode];
            bp.Wweight = new double[Znode, Ynode];

            //Step 0 Initilize random weight
            initializeWeight(bp.Inode,bp.Wweight, bp.Vweight, bp.inputNode, Znode, Ynode);

            bp.x = new double[bp.Inode];
            bp.deltaKError = new double[Ynode];
            bp.Z_in = new double[Znode];
            bp.Z = new double[Znode];
            bp.Y_in = new double[Ynode];
            bp.Y = new double[Ynode];
            bp.weightChangeV = new double[bp.Inode, Znode];
            bp.weightChangeW = new double[Znode, Ynode];
            bp.sum2 = 0.0;

            double benchmark = (1.0 / numberOfClasses) / 2.0;
            int[,] confusionMatrix = new int[(int)numberOfClasses, (int)numberOfClasses];
            while (bp.num <= epoch)
            {
                correctnessIrisSetosa = 0;
                correctnessIrisVersicolour = 0;
                correctnessIrisVirginica = 0;
                incorrectnessIrisSetosa = 0;
                incorrectnessIrisVersicolour = 0;
                incorrectnessIrisVirginica = 0;
                for (int p = 0; p < bp.totalData; p++)
                {
                    for (int i = 0; i < bp.Inode; i++)
                    {
                        bp.x[i] = bp.data[p, i];
                    }
                    
                    feedForward(bp.Inode, Znode, Ynode, bp.x, bp.Vweight, bp.Wweight, bp.Z_in, bp.Z, bp.Y_in, bp.Y, bp.target, p, numberOfClasses, benchmark, confusionMatrix);
                    bp.sum2 = bp.sum2 + backpropagationOfError(bp.Inode, Znode, Ynode, bp.deltaKError, bp.Vweight, bp.Wweight, bp.Y, bp.target, p, learningRate, bp.Z, bp.weightChangeW, bp.weightChangeV, bp.x);
                    updateWeight(bp.weightChangeV, bp.weightChangeW, bp.Vweight, bp.Wweight, bp.Inode, Znode, Ynode);
                    
                }
            /*   if (bp.meanSquareError < 0.1)
                {
                    break;
                }*/
                bp.num++;
    }
            File.Create("C:/Users/Firdaus Samad/Documents/Visual Studio 2015/Projects/backpropagation2/backpropagation2/WeightV.txt").Close();
            File.Create("C:/Users/Firdaus Samad/Documents/Visual Studio 2015/Projects/backpropagation2/backpropagation2/WeightW.txt").Close();
            WriteTextWeight(bp.Vweight, bp.Inode, Znode, "C:/Users/Firdaus Samad/Documents/Visual Studio 2015/Projects/backpropagation2/backpropagation2/WeightV.txt");
            WriteTextWeight(bp.Wweight, Znode, Ynode, "C:/Users/Firdaus Samad/Documents/Visual Studio 2015/Projects/backpropagation2/backpropagation2/WeightW.txt");
            Console.WriteLine("===============================================");
            Console.WriteLine("Confusion Matrix");
            Console.WriteLine("===============================================");
            Console.WriteLine("a\tb\tc");
            for (int i = 0; i < ((int)numberOfClasses); i++)
            {
                for (int y = 0; y < numberOfClasses; y++)
                {
                    Console.Write(confusionMatrix[i, y] + "\t");
                }
                switch (i)
                {
                    case 0:
                        Console.WriteLine("a - Iris-Setosa");
                        break;
                    case 1:
                        Console.WriteLine("b - Iris Versicolour");
                        break;
                    case 2:
                        Console.WriteLine("c - Iris-Virginica");
                        break;
                }
            }
            Console.WriteLine();

            Console.WriteLine("===============================================");
            //for()
            double correctness = correctnessIrisVirginica + correctnessIrisVersicolour + correctnessIrisSetosa;
            double CorrectnessPercent = (correctness / bp.totalData) * 100;
            double NotCorrectnessPercent = 100 - CorrectnessPercent;
            Console.WriteLine("Correctly Classified Instances\t\t:" + (int)correctness + "\t" + CorrectnessPercent + "%");
            Console.WriteLine("Incorrectly Classified Instances\t:" + (bp.totalData - (int)correctness) + "\t" + NotCorrectnessPercent + "%");

            // calAccuracy(numberOfClasses, bp.Y, Ynode, bp.target, bp.totalData);
            bp.meanSquareError = bp.sum2 / epoch;

            Console.WriteLine("MSE :" + bp.meanSquareError);

        }
        //Initialize the weight with random value
        private static void initializeWeight(int Inode, double[,] Wweight, double[,] Vweight, int inputNode, int Znode, int Ynode)
        {
               Random r = new Random();
                    for (int i = 0; i < Inode; i++)
                    {
                        for (int j = 0; j < Znode; j++)
                        {
                            Vweight[i,j] = Convert.ToDouble(r.Next(1, 9) / 10M);
                          // Console.WriteLine(Vweight[i, j]);
                        }
                    }

                    for (int j = 0; j < Znode; j++)
                    {
                        for (int k = 0; k < Ynode; k++)
                        {
                            Wweight[j, k] = Convert.ToDouble(r.Next(1, 9) / 10M);
                        }
                    }
        }
        //Feedforward
        private static void feedForward(int Inode, int Znode, int Ynode, double[] x, double[,] Vweight, double[,] Wweight, double[] Z_in, double[] Z, double[] Y_in, double[] Y, double[] target, int p, double numberOfClasses, double benchmark, int[,] confusionMatrix)
        {
            double sum = 0;
            
           for (int j = 0; j < Znode; j++)
            {
                for (int i = 0; i < Inode; i++)
                {
                    sum += (x[i] * Vweight[i, j]);
                    
                    
                }
                Z_in[j] = sum;
               
                sum = 0;
                Z[j] = 1 / (1 + Math.Exp(-1 * Z_in[j]));
               
            }

            for (int k = 0; k < Ynode; k++)
            {
                for (int j = 0; j < Znode; j++)
                {
                    sum += (Z[j] * Wweight[j, k]);
                    
                }
                Y_in[k] = sum;
               
                sum = 0;
                Y[k] = 1 / (1 + Math.Exp(-1 * Y_in[k]));
               

            }

           
            for (int k = 0; k < Ynode; k++)
            {
       

                if (Math.Abs(0.33 - Y[k]) <= benchmark)
                {


                    if (0.33 == target[p])
                    {

                        // print Correctness
                        correctnessIrisSetosa++;
                        confusionMatrix[0, 0]++;

                    }
                    else {

                        //print incorrectness
                        if (target[p] == 0.67)
                        {
                            confusionMatrix[1, 0]++;
                        }
                        else if (target[p] == 1.0)
                        {
                            confusionMatrix[2, 0]++;
                        }
                        incorrectnessIrisSetosa++;
                    }
                }
                else if (Math.Abs(0.67 - Y[k]) <= benchmark)
                {
                    if (0.67 == target[p])
                    {
                        // print Correctness
                        correctnessIrisVersicolour++;
                        confusionMatrix[1, 1]++;
                    }
                    else {
                        // print inCorrectness
                        if (target[p] == 0.33)
                        {
                            confusionMatrix[0, 1]++;
                        }
                        else if (target[p] == 1.0)
                        {
                            confusionMatrix[2, 1]++;
                        }
                        incorrectnessIrisVersicolour++;
                    }
                }
                else if (Math.Abs(1 - Y[k]) <= benchmark)
                {
                    if (1.0 == target[p])
                    {
                        // print Correctness
                        correctnessIrisVirginica++;
                        confusionMatrix[2, 2]++;
                    }
                    else {
                        // print inCorrectness
                        if (target[p] == 0.67)
                        {
                            confusionMatrix[1, 2]++;
                        }
                        else if (target[p] == 0.33)
                        {
                            confusionMatrix[0, 2]++;
                        }
                        incorrectnessIrisVirginica++;
                    }
                }
            }

        }
        //Backpropagation of Error
        private static double backpropagationOfError(int Inode, int Znode, int Ynode, double[] deltaKError, double[,] Vweight, double[,] Wweight, double[] Y, double[] target, int p, double learningRate, double[] Z, double[,] weightChangeW, double[,] weightChangeV, double[] x)
        {
            double[] deltaJ_in = new double[Znode];
            double[] deltaJ = new double[Znode];
            double sum2 = 0.0;
            //Calculate error 
            for (int k = 0; k < Ynode; k++)
            {

                deltaKError[k] = (target[p] - Y[k]) * Y[k] * (1 - Y[k]);
                sum2 += Math.Pow(deltaKError[k], 2);
            }
            //calculate weight change for w

            for (int k = 0; k < Ynode; k++)
                {
                for (int j = 0; j < Znode; j++)
                {

                    weightChangeW[j, k] = learningRate * deltaKError[k] * Z[j];
                }
                
            }
            //Calculate Delta J_in
            for (int k = 0; k < Ynode; k++)
            {
                for (int j = 0; j < Znode; j++)
            {
               
                    deltaJ_in[j] = deltaJ_in[j] + deltaKError[k] * Wweight[j, k];
                }
            }

            //Calculate delta J
            for (int j = 0; j < Znode; j++)
            {
                deltaJ[j] = deltaJ_in[j] * Z[j] * (1 - Z[j]);
            }
            //Weight Change v
            
                for (int j = 0; j < Znode; j++)
                {
                for (int i = 0; i < Inode; i++)
                {
                    weightChangeV[i, j] = learningRate * deltaJ[j] * x[i];
                }
            }

            return sum2;
        }
        // update the weight
        private static void updateWeight(double[,] weightChangeV, double[,] weightChangeW, double[,] Vweight, double[,] Wweight, int Inode, int Znode, int Ynode)
        {

            for (int j = 0; j < Znode; j++)
            {
                for (int i = 0; i < Inode; i++)
                {
                    Vweight[i, j] = Vweight[i, j] + weightChangeV[i, j];
                }
            }
            for (int k = 0; k < Ynode; k++)
            {
                for (int j = 0; j < Znode; j++)
                {
                    Wweight[j, k] = Wweight[j, k] + weightChangeW[j, k];    
                } 
            }
        }
        //Testing
        public void testing(string path, int Znode, int Ynode, double learningRate, double numberOfClasses)
        {
            Console.WriteLine("\n\nTesting Result :");
            bpnn bp = new bpnn();
            bp.num = 1;
            bp.inputNode = 0;
            bp.totalData = 0;

            //read data from txt file
            using (StreamReader myReader = new StreamReader(path))
            {
                string line = " ";
                string l = "";
                while (line != null)
                {
                    line = myReader.ReadLine();
                    if (line != null)
                    {
                        l = line;
                        //Split data by comma
                        String[] value = l.Split(null);

                        //to get the number of input node
                        bp.inputNode = value.Length;
                        //to get the number of total iteration/ line of data(150 data)
                        bp.totalData = bp.totalData + 1;
                    }
                }
                myReader.Close();
            }
            //Read Input data
            bp.Inode = bp.inputNode - 1;
            bp.data = new double[bp.totalData, bp.Inode];
            bp.target = new double[bp.totalData];
            using (StreamReader myReader = new StreamReader(path))
            {
                string l = "";
                int a = 0;
                string line = myReader.ReadLine();
                while (line != null)
                {
                    l = line;
                    //Split data by space
                    String[] value = l.Split(null);

                    for (int i = 0; i < bp.inputNode; i++)
                    {
                        if (i == bp.inputNode - 1)
                            bp.target[a] = Convert.ToDouble(value[i]);
                        else {
                            bp.data[a, i] = Convert.ToDouble(value[i]);
                        }
                    }
                    a++;

                    line = myReader.ReadLine();
                }
                myReader.Close();
            }
            bp.Vweight = new double[bp.Inode, Znode];
            bp.Wweight = new double[Znode, Ynode];

            bp.x = new double[bp.Inode];
            bp.deltaKError = new double[Ynode];
            bp.Z_in = new double[Znode];
            bp.Z = new double[Znode];
            bp.Y_in = new double[Ynode];
            bp.Y = new double[Ynode];
            bp.weightChangeV = new double[bp.Inode, Znode];
            bp.weightChangeW = new double[Znode, Ynode];
            bp.sum2 = 0.0;

            double benchmark = (1.0 / numberOfClasses) / 2.0;
            int[,] confusionMatrix = new int[(int)numberOfClasses, (int)numberOfClasses];
            readWeight(Znode, bp.Inode, Ynode, bp.Wweight, bp.Vweight);
                correctnessIrisSetosa = 0;
                correctnessIrisVersicolour = 0;
                correctnessIrisVirginica = 0;
                incorrectnessIrisSetosa = 0;
                incorrectnessIrisVersicolour = 0;
                incorrectnessIrisVirginica = 0;
                for (int p = 0; p < bp.totalData; p++)
                {
                    for (int i = 0; i < bp.Inode; i++)
                    {
                        bp.x[i] = bp.data[p, i];
                    }

                    feedForward(bp.Inode, Znode, Ynode, bp.x, bp.Vweight, bp.Wweight, bp.Z_in, bp.Z, bp.Y_in, bp.Y, bp.target, p, numberOfClasses, benchmark, confusionMatrix);

                }
                /*   if (bp.meanSquareError < 0.1)
                    {
                        break;
                    }*/
                bp.num++;
            Console.WriteLine("===============================================");
            Console.WriteLine("Confusion Matrix");
            Console.WriteLine("===============================================");
            Console.WriteLine("a\tb\tc");
            for (int i = 0; i < ((int)numberOfClasses); i++)
            {
                for (int y = 0; y < numberOfClasses; y++)
                {
                    Console.Write(confusionMatrix[i, y] + "\t");
                }
                switch (i)
                {
                    case 0:
                        Console.WriteLine("a - Iris-Setosa");
                        break;
                    case 1:
                        Console.WriteLine("b - Iris Versicolour");
                        break;
                    case 2:
                        Console.WriteLine("c - Iris-Virginica");
                        break;
                }
            }
            Console.WriteLine();

            Console.WriteLine("===============================================");
            //for()
            double correctness = correctnessIrisVirginica + correctnessIrisVersicolour + correctnessIrisSetosa;
            double CorrectnessPercent = (correctness / bp.totalData) * 100;
            double NotCorrectnessPercent = 100 - CorrectnessPercent;
            Console.WriteLine("Correctly Classified Instances\t\t:" + (int)correctness + "\t" + CorrectnessPercent + "%");
            Console.WriteLine("Incorrectly Classified Instances\t:" + (bp.totalData - (int)correctness) + "\t" + NotCorrectnessPercent + "%");

        }
        //Read weight from text WeightW.txt and WeightV.txt
        public static void readWeight(int Znode,int Inode, int Ynode,double [,]Wweight,double[,]Vweight)
        {
            using (StreamReader read = new StreamReader("C:/Users/Firdaus Samad/Documents/Visual Studio 2015/Projects/backpropagation2/backpropagation2/WeightV.txt", true)) 
            {
                
                    int o = 0;

                    string line = "";
                    line = read.ReadLine();

                       
                    while (line != null && o < Inode)
                    {
                    for (int q = 0; q < Znode; q++)
                    {

                        Vweight[o,q] = Convert.ToDouble(line);
                        line = read.ReadLine();
                        }
                    o++;
                }
                o = 0;


                read.Close();
                
        }

            using (StreamReader read1 = new StreamReader("C:/Users/Firdaus Samad/Documents/Visual Studio 2015/Projects/backpropagation2/backpropagation2/WeightW.txt", true))
            {

                int o = 0;

                string line = "";
                line = read1.ReadLine();

                 
                    while (line != null && o < Znode)
                    {
                    for (int q = 0; q < Ynode; q++)
                    {

                        Wweight[o, q] = Convert.ToDouble(line);
                        line = read1.ReadLine();
                        
                    }
                    o++;
                }

                o = 0;
                read1.Close();

            }

        }
        //Write weight to txt file 
        public static void WriteTextWeight(double[,] weight, int Iarray, int Jarray, string path)
        {
            using (StreamWriter write = new StreamWriter(path, true))
            {
                for (int i = 0; i < Iarray; i++)
                {
                    for (int j = 0; j < Jarray; j++)
                    {
                        if (i == 0 && j == 0)
                            write.Write(weight[i, j]);
                        else write.Write("\n{0}", weight[i, j]);
                    }
                }
                write.Close();
            }
        }
    }
}










