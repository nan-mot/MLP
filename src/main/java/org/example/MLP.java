package org.example;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class MLP {
    
    private final int epochCount;
    private final double learningRate;


    private Double[] hiddenLayerResults;
    private Double[][] inputToHiddenLayerWeights;

    private Double[] outerLayerResults;
    private Double[][] hiddenToOuterLayerWeights;


    public MLP(int epochCount,
               double learningRate,
               int hiddenLayerSize,
               int outerLayerSize,
               Double[][] trainingSet, Double[] trainingAnswer) throws IOException {
        this.epochCount = epochCount;
        this.learningRate = learningRate;
        initializationWeight(trainingSet[0].length, hiddenLayerSize, outerLayerSize);
        study(trainingSet, trainingAnswer);
    }


    public Double[] calculateResult(Double[] input) {
        hiddenLayerResults = calculateLayerNodeResults(input, inputToHiddenLayerWeights);
        return calculateLayerNodeResults(hiddenLayerResults, hiddenToOuterLayerWeights);
    }


    private void initializationWeight(int inputSize, int hiddenLayerSize, int outerLayerSize) {
        inputToHiddenLayerWeights = new Double[hiddenLayerSize][inputSize];
        for (int i = 0; i < inputToHiddenLayerWeights.length; i++) {
            for (int j = 0; j < inputToHiddenLayerWeights[i].length; j++) {
                inputToHiddenLayerWeights[i][j] = Math.random();
            }
        }
        hiddenToOuterLayerWeights = new Double[outerLayerSize][hiddenLayerSize];
        for (int i = 0; i < hiddenToOuterLayerWeights.length; i++) {
            for (int j = 0; j < hiddenToOuterLayerWeights[i].length; j++) {
                hiddenToOuterLayerWeights[i][j] = Math.random();
            }
        }
    }

    private void study(Double[][] trainingSet, Double[] trainingAnswer) throws IOException {
        Double[] hiddenDelta;
        Double[] outerDelta;
        Double errorFunction = 0D;
        File errorFile = new File("./gError.txt");
        if (errorFile.exists()) {
            errorFile.delete();
        }
        Path errorFilePath = errorFile.toPath();
        StringBuilder builder = new StringBuilder();
        for (int e = 0; e < epochCount; e++) {
            for (int n = 0; n < trainingSet.length; n++) {
                hiddenLayerResults = calculateLayerNodeResults(trainingSet[n], inputToHiddenLayerWeights);
                outerLayerResults = calculateLayerNodeResults(hiddenLayerResults, hiddenToOuterLayerWeights);

                errorFunction = calculateErrorFunction(new Double[]{trainingAnswer[n]}, outerLayerResults);
                builder.append(errorFunction).append(System.lineSeparator());

                outerDelta = outerLevelDeltas(outerLayerResults, new Double[]{trainingAnswer[n]}, hiddenLayerResults, hiddenToOuterLayerWeights);
                hiddenToOuterLayerWeights = recalculateLayerWeights(hiddenLayerResults, outerDelta, hiddenToOuterLayerWeights);

                hiddenDelta = hiddenLevelDeltas(hiddenLayerResults, hiddenToOuterLayerWeights, outerDelta, trainingSet[n], inputToHiddenLayerWeights);
                inputToHiddenLayerWeights = recalculateLayerWeights(trainingSet[n], hiddenDelta, inputToHiddenLayerWeights);
            }
        }
        Files.write(errorFilePath, builder.toString().getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        System.out.println("Error function: " + errorFunction);
    }


    /**
     * Вычисление значений ВСЕХ узлов слоя.
     *
     * @param inputValues - значения предыдущего слоя
     * @param weights - веса для к
     * @return вычисленное значение узла
     */
    private Double[] calculateLayerNodeResults(Double[] inputValues, Double[][] weights) {
        Double[] layerNodeResults = new Double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            layerNodeResults[i] = calculateNodeResult(inputValues, weights[i]);
        }
        return layerNodeResults;
    }

    /**
     * Вычисление значения ОДНОГО узла нейронной сети
     * inputValues - значения предыдущего слоя
     *
     * @param inputValues - значения предыдущего слоя
     * @param weights - веса
     * @return вычисленное значение узла
     */
    private Double calculateNodeResult(Double[] inputValues, Double[] weights) {
        double sum = 0D;
        for (int i = 0; i < inputValues.length; i++) {
            sum += inputValues[i] * weights[i];
        }
        return activationFunction(sum);
    }

    private Double calculateErrorFunction(Double[] targetValues, Double[] resultValues) {
        double errorFunction = 0D;
        double toPow;
        double powResult;
        for (int k = 0; k < targetValues.length; k++) {
            toPow = Math.abs(targetValues[k] - resultValues[k]);
            powResult = Math.pow(toPow, 2D);
            errorFunction += powResult;
        }
        return errorFunction / 2;
    }

    private Double coefficient = 0.005;

    private Double activationFunction(Double netj) {
        //return coefficient * netj;
        return 1 / (1 + Math.exp( netj * coefficient));
    }

    private Double activationFunctionDerivative(Double x) {
        //return coefficient;
        return (1 - activationFunction(x)) * activationFunction(x);
    }


    private Double[] outerLevelDeltas(Double[] outValues, Double[] answers, Double[] inputValues, Double[][] weights) {
        Double[] outerLevelDeltas = new Double[outValues.length];
        for (int i = 0; i < outerLevelDeltas.length; i++) {
            outerLevelDeltas[i] = outerLevelSingleDelta(outValues[i], answers[i], inputValues, weights[i]);
        }
        return outerLevelDeltas;
    }

    private Double outerLevelSingleDelta(Double outValue, Double expectedValue, Double[] inputValues, Double[] weights) {
        return (expectedValue - outValue) * activationFunctionDerivative(calculateNet(inputValues, weights));
    }

    private Double calculateNet(Double[] inputValues, Double[] weights) {
        double sum = 0D;
        for (int i = 0; i < inputValues.length; i++) {
            sum += inputValues[i] * weights[i];
        }
        return sum;
    }

    private Double[] hiddenLevelDeltas(Double[] hiddenLevelValues, Double[][] hiddenToOuterWeights,
                                       Double[] outerDeltas, Double[] inputValues, Double[][] inputToHiddenWeights) {
        Double[] hiddenLevelDeltas = new Double[hiddenLevelValues.length];
        Double sum;
        for (int j = 0; j < hiddenLevelDeltas.length; j++) {
            sum = 0D;
            for (int k = 0; k < outerDeltas.length; k++) {
                sum += outerDeltas[k] * hiddenToOuterWeights[k][j];
            }
            hiddenLevelDeltas[j] = sum * activationFunctionDerivative(calculateNet(inputValues, inputToHiddenWeights[j]));
        }
        return hiddenLevelDeltas;
    }

    private Double[][] recalculateLayerWeights(Double[] inputResults, Double[] deltas, Double[][] currentWeights) {
        Double[][] previousLayerNewWeights = new Double[currentWeights.length][currentWeights[0].length];
        for (int i = 0; i < deltas.length; i++) {
            for (int j = 0; j < previousLayerNewWeights[0].length; j++) {
                previousLayerNewWeights[i][j] = currentWeights[i][j] + learningRate * deltas[i] * inputResults[j];
            }
        }
        return previousLayerNewWeights;
    }
}
