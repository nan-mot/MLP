package org.example;

public class MLP {
    
    private final int epochCount;
    private final double learningRate;


    private Double[] hiddenLayerVector;
    private Double[][] inputToHiddenLayerWeights;

    private Double[] outerLayerVector;
    private Double[][] hiddenToOuterLayerWeights;


    public MLP(int epochCount,
               double learningRate,
               int hiddenLayerSize,
               int outerLayerSize,
               Double[][] trainingSet, Double[] trainingAnswer) {
        this.epochCount = epochCount;
        this.learningRate = learningRate;
        initializationWeight(trainingSet[0].length, hiddenLayerSize, outerLayerSize);
        study(trainingSet, trainingAnswer);
    }

    private void initializationWeight(int inputSize, int hiddenLayerSize, int outerLayerSize) {
        inputToHiddenLayerWeights = new Double[inputSize][hiddenLayerSize];
        for (int i = 0; i < inputToHiddenLayerWeights.length; i++) {
            for (int j = 0; j < inputToHiddenLayerWeights[i].length; j++) {
                inputToHiddenLayerWeights[i][j] = Math.random();
            }
        }
        hiddenToOuterLayerWeights = new Double[hiddenLayerSize][outerLayerSize];
        for (int i = 0; i < hiddenToOuterLayerWeights.length; i++) {
            for (int j = 0; j < hiddenToOuterLayerWeights[i].length; j++) {
                hiddenToOuterLayerWeights[i][j] = Math.random();
            }
        }
    }

    private void study(Double[][] trainingSet, Double[] trainingAnswer) {
        Double[] delta;
        Double[] inaccuracyVector;
        for (int e = 0; e < epochCount; e++) {
            for (int n = 0; n < trainingSet.length; n++) {
                hiddenLayerVector = countNextLayerVector(trainingSet[n], inputToHiddenLayerWeights);
                outerLayerVector = countNextLayerVector(hiddenLayerVector, hiddenToOuterLayerWeights);
                delta = calculateDelta(new Double[]{trainingAnswer[n]}, outerLayerVector);
                inaccuracyVector = calculateLayerInaccuracyVector(hiddenLayerVector, delta, hiddenToOuterLayerWeights);
                inputToHiddenLayerWeights = recalculateLayerWeights(inputToHiddenLayerWeights, trainingSet[n], inaccuracyVector);
                hiddenToOuterLayerWeights = recalculateLayerWeights(hiddenToOuterLayerWeights, hiddenLayerVector, delta);
            }
        }
    }

    public Double[] calculateResult(Double[] input) {
        hiddenLayerVector = countNextLayerVector(input, inputToHiddenLayerWeights);
        return countNextLayerVector(hiddenLayerVector, hiddenToOuterLayerWeights);
    }


    /////////////////////////////////////////////////////////////////////////////////////


    private Double[] countNextLayerVector(Double[] previousLayerVector, Double[][] weights) {
        double sum;
        Double[] layerVector = new Double[weights[0].length];
        for (int j = 0; j < weights[0].length; j ++) {
            sum = 0;
            for (int i = 0; i < previousLayerVector.length; i++) {
                sum += previousLayerVector[i] * weights[i][j];
            }
            layerVector[j] = activationFunction(sum);
        }
        return layerVector;
    }

    private Double[] calculateDelta(Double[] targetVector, Double[] resultVector) {
        Double[] deltaVector = new Double[targetVector.length];
        for (int k = 0; k < targetVector.length; k++) {
            deltaVector[k] = inaccuracyFunction(targetVector[k], resultVector[k]);
        }
        return deltaVector;
    }

    private Double[] calculateLayerInaccuracyVector(Double[] layerValues, Double[] deltaVector, Double[][] nextLayerWeights) {
        Double[] inaccuracyVector = new Double[layerValues.length];
        Double sum;
        for (int j = 0; j < layerValues.length; j++) {
            sum = 0D;
            for (int k = 0; k < deltaVector.length; k++) {
                sum += deltaVector[k] * nextLayerWeights[j][k];
            }
            inaccuracyVector[j] = layerValues[j] * (1 - layerValues[j]) * sum;
        }
        return inaccuracyVector;
    }

    private Double[][] recalculateLayerWeights(Double[][] weights, Double[] previousLayerVector, Double[] layerInaccuracyVector) {
        Double[][] recalculatedWeights = new Double[weights.length][weights[0].length];
        for (int i = 0; i < previousLayerVector.length; i++) {
            for (int j = 0; j < layerInaccuracyVector.length; j++) {
                recalculatedWeights[i][j] = weights[i][j] + learningRate * previousLayerVector[i] * layerInaccuracyVector[j];
            }
        }
        return recalculatedWeights;
    }

    private Double coefficient = 0.005;

    private Double activationFunction(Double netj) {
        //activation function
        //return (1 / (1 + Math.exp(0 - netj)));
        return coefficient * netj;
    }

    private Double activationFunctionDerivative(Double x) {
        //return x * (1 - x);
        return coefficient;
    }

    private Double inaccuracyFunction(Double targetValue, Double resultValue) {
        //inaccuracy function
        //return (targetValue - resultValue) * resultValue * (1 - resultValue);
        return (targetValue - resultValue) * activationFunctionDerivative(resultValue);
    }


    /////////////////////////////////////////////////////////


 /*

     double[] hidden;
    double[][] wInputHidden;
    double[] wHiddenOuter;

     public void initializationWeight() {
        for (int i = 0; i < wInputHidden.length; i++) {
            for (int j = 0; j < wInputHidden[i].length; j++) {
                wInputHidden[i][j] = Math.random() * 0.1 + 0.1;
            }
        }
        for (int i = 0; i < wHiddenOuter.length; i++) {
            wHiddenOuter[i] = Math.random() * 0.1 + 0.1;
        }
    }


 public double countOuter(Double[] input) {

        for (int i = 0; i < )

            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = 0;
                for (int j = 0; j < input.length; j++) {
                    hidden[i] += input[j] * wInputHidden[j][i];
                }
                hidden[i] = activationFunction(hidden[i]);
            }

        double outer = 0;

        for (int i = 0; i < hidden.length; i++) {
            outer += hidden[i] * wHiddenOuter[i];
        }

        //activation function
        return activationFunction(outer);
//        if ((1 / (1+Math.exp(-outer))) > 0.51) outer = 1; else outer = 0;
    }


    public void study() {
        double[] err = new double[hidden.length];
        double gError = 0;
        Double[] input = new Double[trainingSet[0].length];
        double outer;

        for (int i = 0; i < epochCount; i ++) {
            gError = 0;
            for (int j = 0; j < trainingSet.length; j++) {

                System.arraycopy(trainingSet[j], 0, input, 0, input.length);

                outer = countOuter(input);

                //FIXME исправить формулу вычисления ошибки
                double lError = outer - trainingAnswer[j];
                gError += (Math.pow(lError, 2)) / trainingAnswer.length;

                for (int k = 0; k < hidden.length; k++) {
                    err[k] = lError * wHiddenOuter[k];
                }

                for (int k = 0; k < input.length; k++) {
                    for (int v = 0; v < hidden.length; v++) {
                        //FIXME исправить формулу
                        wInputHidden[k][j] += 0.1 * err[j] * input[i]; //обуч коэф * знач ошибки скрытом слое * н  а значение входного нейрона
                    }
                }

                for (int i = 0; i < hidden.length; i++) {
                    //FIXME исправить формулу
                    wHiddenOuter[i] += 0.1 * lError * hidden[i]; //корр весов на выходном
                }
            }
        }
        System.out.println("gError = " + gError);
       // System.out.println("Epochs = " + count);
    }*/
}
