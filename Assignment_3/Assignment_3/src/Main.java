import java.util.Random;

/**
 * @author Harmandeep Mangat || hm15mx || 6021109
 *
 * Running Instructions:
 * Takes 4 command Line Arugments;
 * 1. Number of input nodes
 * 2. Number of hidden Nodes
 * 3. Number of output nodes
 * 4. learning rate
 */
public class Main {
    /**
     * trains a feed forward neural network using the backpropagation algorithm
     * @param numberOfInputNodes
     * @param numberOfHiddenNodes
     * @param numberOfOutputNodes
     * @param learningRate
     */
    public Main(int numberOfInputNodes, int numberOfHiddenNodes, int numberOfOutputNodes,
                double learningRate) {
       int epoch = 10000;
       int iterationNumber = 0;
       double error = 0;

       int[][] trainingData = generateTrainingData();
       Node[] inputNodes = createInputNodes(numberOfInputNodes);
       Node[] hiddenNodes = createHiddenNodes(numberOfHiddenNodes);
       Node[] outputNodes = createOutputNodes(numberOfOutputNodes);
       double[][] weight1 = generateWeightsZeroOne(numberOfInputNodes,numberOfHiddenNodes);
       double[][] weight2 = generateWeightsOneTwo(numberOfHiddenNodes,numberOfOutputNodes);
       Random random = new Random();

        for (int i = 0; i < epoch; i++) {
            for (int k = 0; k < 16; k++) {
                int index1 = random.nextInt(trainingData.length);
                for (int l = 0; l < inputNodes.length; l++) {
                    inputNodes[l].setA(trainingData[index1][l]);
                }
                // calculate s and activation from layer 0 to 1
                for (int j = 0; j < numberOfHiddenNodes; j++) {
                    hiddenNodes[j].setS(calculateSZeroOne(weight1,inputNodes,j));
                    hiddenNodes[j].setA(calculateActivation(hiddenNodes[j].getS()));
                }

                // calculate s and activation from layer 1 to 2
                for (int j = 0; j < outputNodes.length; j++) {
                    outputNodes[j].setS(calculateSOneTwo(weight2,hiddenNodes,j));
                    outputNodes[j].setA(calculateActivation(outputNodes[j].getS()));
                }

                // calculate error of output nodes and derivative of the error
                for (int j = 0; j < outputNodes.length; j++) {
                    outputNodes[j].setDO(errorOfOutputNode(outputNodes,trainingData,index1,j));
                    error = error + (Math.pow(outputNodes[j].getDO(),2) / 2);
                    outputNodes[j].setDI(derivativeOfError(outputNodes,j));
                }

                // calculate the error of the hidden nodes and derivative of the error
                for (int j = 0; j < hiddenNodes.length; j++) {
                    hiddenNodes[j].setDO(errorOfHiddenNodes(outputNodes,weight2,j));
                    hiddenNodes[j].setDI(derivativeOfError(hiddenNodes,j));
                }

                // update weights
                for (int j = 0; j < weight1.length; j++) {
                    for (int l = 0; l < numberOfInputNodes; l++) {
                        // update weights connecting input layer to hidden layer
                        weight1[j][l] = weight1[j][l] - changeInWeight(hiddenNodes,inputNodes,learningRate,l, j);
                    }
                }

                for (int j = 0; j < weight2.length; j++) {
                    for (int l = 0; l < numberOfOutputNodes; l++) {
                        weight2[j][l] = weight2[j][l] - changeInWeight(outputNodes,hiddenNodes,learningRate,l, j);
                    }
                }
            }
            iterationNumber ++;
            if (iterationNumber==200) {
                iterationNumber = 0;
                double meanSquaredError = error / (16*2);
                System.out.println("Epoch: " + i + "   Mean Squared Error: " + meanSquaredError);
            }
            if (i < epoch-1) {
                error = 0;
            }
        }
        System.out.println("Number of Hidden Nodes: " + numberOfHiddenNodes);
        System.out.println("Learning Rate: " + learningRate);
        double meanSquaredError = error / (16 * 2);
        System.out.println("Mean Squared Error: " + meanSquaredError);
        System.out.println();

        // after training
        for (int i = 0; i < 16; i++) {
            int index1 = random.nextInt(trainingData.length);
            for (int j = 0; j < numberOfInputNodes; j++) {
                inputNodes[j].setA(trainingData[index1][j]);
                System.out.print(trainingData[index1][j] + "  ");
            }
            for (int j = 0; j < numberOfHiddenNodes; j++) {
                hiddenNodes[j].setS(calculateSZeroOne(weight1,inputNodes,j));
                hiddenNodes[j].setA(calculateActivation(hiddenNodes[j].getS()));
            }

            // calculate s and activation from layer 1 to 2
            for (int j = 0; j < outputNodes.length; j++) {
                outputNodes[j].setS(calculateSOneTwo(weight2,hiddenNodes,j));
                outputNodes[j].setA(calculateActivation(outputNodes[j].getS()));
            }
            System.out.print("Outcome: " + outputNodes[0].getA() + "  Expected Outcome: " +
                    trainingData[index1][trainingData[1].length-1]);
            System.out.println();
        }
    }

    /**
     * calculates the activation
     * @param s
     * @return
     */
    private double calculateActivation(double s) {
        return 1/(1 + Math.exp(-s));
    }

    /**
     * calculates the summation of weights * activation for one hidden node
     * @param weight1
     * @param inputNodes
     * @param index
     * @return
     */
    private double calculateSZeroOne(double[][] weight1, Node[] inputNodes, int index) {
        double s = 0;

        for (int i = 0; i < weight1.length; i++) {
            s = s + (weight1[i][index]*inputNodes[i].getA());
        }
        return s;
    }

    /**
     * calculates the summation of weights * activation for one output node
     * @param weight2
     * @param hiddenNodes
     * @param index
     * @return
     */
    private double calculateSOneTwo(double[][] weight2, Node[] hiddenNodes, int index) {
        double s = 0;

        for (int i = 0; i < weight2.length; i++) {
            s = s + (weight2[i][index]*hiddenNodes[i].getA());
        }
        return s;
    }

    /**
     * calculates the error for one output node
     * @param outputNodes
     * @param trainingData
     * @param index location of the expected outcome
     * @param index2 location of the output node
     * @return
     */
    private double errorOfOutputNode(Node[] outputNodes, int[][] trainingData, int index, int index2) {
        return outputNodes[index2].getA() - trainingData[index][trainingData[1].length-1];
    }

    /**
     * calculates the derivative of the error
     * @param currentNode
     * @param index
     * @return
     */
    private double derivativeOfError(Node[] currentNode, int index) {
        return currentNode[index].getDO() * (calculateActivation(currentNode[index].getS()) *
                (1-calculateActivation(currentNode[index].getS())));

    }

    /**
     * calculates the error of the hidden nodes
     * @param outputNodes
     * @param weight2
     * @param index
     * @return
     */
    private double errorOfHiddenNodes(Node[] outputNodes, double[][] weight2, int index) {
        double DO = 0;

        for (int i = 0; i < outputNodes.length; i++) {
            DO = DO + (outputNodes[i].getDI() * weight2[index][i]);
        }
        return DO;
    }

    /**
     * calculates the change in weight
     * @param nextLayer
     * @param currentLayer
     * @param learningRate
     * @param index
     * @param index2
     * @return
     */
    private double changeInWeight(Node[] nextLayer, Node[] currentLayer, double learningRate, int index,
                                  int index2) {
        return nextLayer[index].getDI() * currentLayer[index2].getA() * learningRate;
    }

    /**
     * creates n amount of input nodes, where n>0
     * @param n number of input nodes
     */
    private Node[] createInputNodes(int n) {
        Node[] inputNodes = new Node[n];

        for (int i = 0; i < n; i++) {
            inputNodes[i] = new Node();
        }

        return inputNodes;
    }

    /**
     * creates n amount of hidden nodes
     * @param n
     * @return
     */
    private Node[] createHiddenNodes(int n) {
        Node[] hiddenNodes = new Node[n];

        for (int i = 0; i < n; i++) {
            hiddenNodes[i] = new Node();
        }

        return hiddenNodes;
    }

    /**
     * creates n amount of output nodes
     * @param n
     * @return
     */
    private Node[] createOutputNodes(int n) {
        Node[] outputNodes = new Node[n];

        for (int i = 0; i < n; i++) {
            outputNodes[i] = new Node();
        }

        return outputNodes;
    }

    /**
     * randomly generates weights connecting layer zero to one
     * @return
     */
    private double[][] generateWeightsZeroOne(int numberOfInputs, int numberOfHidden) {
        double[][] weight = new double[numberOfInputs][numberOfHidden];
        Random random = new Random();
        for (int i = 0; i < numberOfInputs; i++) {
            for (int j = 0; j < numberOfHidden; j++) {
                double weights = random.nextGaussian();
                while (weights>1 || weights<-1) {
                    weights = random.nextGaussian();
                }
                weight[i][j] = weights;
            }
        }
        return weight;
    }

    /**
     * randomly generates weights connecting layer one to two
     * @param numberOfHidden
     * @param numberOfOutput
     * @return
     */
    private double[][] generateWeightsOneTwo(int numberOfHidden, int numberOfOutput) {
        double[][] weight = new double[numberOfHidden][numberOfOutput];
        Random random = new Random();
        for (int i = 0; i < numberOfHidden; i++) {
            for (int j = 0; j < numberOfOutput; j++) {
                double weights = random.nextGaussian();
                while (weights>1 || weights<-1) {
                    weights = random.nextGaussian();
                }
                weight[i][j] = weights;
            }
        }
        return weight;
    }

    /**
     * randomly generates training data
     * @return
     */
    private int[][] generateTrainingData() {
        int[][] trainingData = new int[4][5];

        for (int i = 0; i < 4; i++) {
            int numberOfOnes = 0;
            for (int j = 0; j < 4; j++) {
                int num =(int)Math.round(Math.random());
                trainingData[i][j] = num;
                if (num==1) {
                    numberOfOnes++;
                }
            }
            if (numberOfOnes%2==0) {
                trainingData[i][4] = 1;
            } else {
                trainingData[i][4] = 0;
            }
        }
        return trainingData;
    }

    public static void main(String[] args) {
        int numberOfInputNodes;
        int numberOfHiddenNodes;
        int numberOfOutputNodes;
        double learningRate;

        try{
            numberOfInputNodes = Integer.parseInt(args[0]);
            numberOfHiddenNodes = Integer.parseInt(args[1]);
            numberOfOutputNodes = Integer.parseInt(args[2]);
            learningRate = Double.parseDouble(args[3]);
            Main m = new Main(numberOfInputNodes,numberOfHiddenNodes,numberOfOutputNodes,learningRate);
        } catch (NumberFormatException e) {
            System.err.println("Input Error");
        }
    }
}
