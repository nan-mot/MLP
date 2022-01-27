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

    private Double[][] inputToHiddenLayerWeights;
    private Double[][] hiddenToOuterLayerWeights;


    public MLP(int epochCount,
               double learningRate,
               int hiddenLayerSize,
               int outerLayerSize,
               Double[][] trainingSet, Double[][] trainingAnswer) throws IOException {
        this.epochCount = epochCount;
        this.learningRate = learningRate;
        initializationWeight(trainingSet[0].length, hiddenLayerSize, outerLayerSize);
        study(trainingSet, trainingAnswer);
    }


    public Double[] calculateResult(Double[] input) {
        Double[] hiddenLayerResults = calculateLayerNodeResults(input, inputToHiddenLayerWeights);
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
        for (int i = 0; i < outerLayerSize; i++) {
            for (int j = 0; j < hiddenToOuterLayerWeights[0].length; j++) {
                hiddenToOuterLayerWeights[i][j] = Math.random();
            }
        }
    }

    private void study(Double[][] trainingSet, Double[][] trainingAnswer) throws IOException {
        Double[] hiddenDelta;
        Double[] outerDelta;
        Double errorFunction = 0D;
        Double[] hiddenLayerResults;
        Double[] outerLayerResult;
        File errorFile = new File("./gError.txt");
        if (errorFile.exists()) {
            errorFile.delete();
        }
        Path errorFilePath = errorFile.toPath();
        StringBuilder builder = new StringBuilder();
        for (int e = 0; e < epochCount; e++) {
            for (int n = 0; n < trainingSet.length; n++) {
                hiddenLayerResults = calculateLayerNodeResults(trainingSet[n], inputToHiddenLayerWeights);
                outerLayerResult = calculateLayerNodeResults(hiddenLayerResults, hiddenToOuterLayerWeights);

                errorFunction = calculateErrorFunction(trainingAnswer[n], outerLayerResult);
                builder.append(errorFunction).append(System.lineSeparator());

                outerDelta = outerLevelDeltas(outerLayerResult, trainingAnswer[n]);
                hiddenToOuterLayerWeights = recalculateLayerToNextWeights(hiddenToOuterLayerWeights, outerDelta, hiddenLayerResults);

                hiddenDelta = hiddenLevelDeltas(outerDelta, hiddenToOuterLayerWeights, hiddenLayerResults);
                inputToHiddenLayerWeights = recalculateLayerToNextWeights(inputToHiddenLayerWeights, hiddenDelta, trainingSet[n]);
            }
        }
        Files.write(errorFilePath, builder.toString().getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        System.out.println("Error function: " + errorFunction);
    }


    /**
     * Вычисление значений ВСЕХ узлов скрытого слоя.
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
     * Вычисление значений ВСЕХ узлов слоя.
     *
     * @param inputValues - значения предыдущего слоя
     * @param weights - веса для к
     * @return вычисленное значение узла
     */
    private Double calculateOuterLayerNodeResult(Double[] inputValues, Double[] weights) {
        return calculateNodeResult(inputValues, weights);
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
        return 1 / (1 + Math.exp(-netj));
    }

    private Double activationFunctionDerivative(Double x) {
        //return coefficient;
        return (1 - activationFunction(x)) * activationFunction(x);
    }


    private Double[] outerLevelDeltas(Double[] outValues, Double[] answers) {
        Double[] outerLevelDeltas = new Double[outValues.length];
        for (int i = 0; i < outerLevelDeltas.length; i++) {
            outerLevelDeltas[i] = (outValues[i] - answers[i]) * outValues[i] * (1 - outValues[i]);
        }
        return outerLevelDeltas;
    }

    private Double[] hiddenLevelDeltas(Double[] previousLayerDeltas, Double[][] hiddenToNextWeights, Double[] outputValues) {
        Double[] hiddenLevelDeltas = new Double[outputValues.length];
        Double[] weightedOuterErrorsSum = weightedOuterErrorsSum(previousLayerDeltas, hiddenToNextWeights);
        for (int i = 0; i < hiddenLevelDeltas.length; i++) {
            hiddenLevelDeltas[i] = weightedOuterErrorsSum[i] * (outputValues[i] * (1 - outputValues[i]));
        }
        return hiddenLevelDeltas;
    }

    private Double[] weightedOuterErrorsSum(Double[] previousLayerDeltas, Double[][] levelToNextWeights) {
        Double[] weightedErrorsSum = new Double[levelToNextWeights[0].length];
        double sum = 0D;
        for (int i = 0; i < weightedErrorsSum.length; i++) {
            for (int j = 0; j < levelToNextWeights.length; j++) {
                sum += previousLayerDeltas[j] * levelToNextWeights[j][i];
            }
            weightedErrorsSum[i] = sum;
        }
        return weightedErrorsSum;
    }

    private Double[][] recalculateLayerToNextWeights(Double[][] currentWeights, Double[] deltas, Double[] inputValues) {
        Double[][] previousLayerNewWeights = new Double[currentWeights.length][currentWeights[0].length];
        for (int i = 0; i < deltas.length; i++) {
            for (int j = 0; j < previousLayerNewWeights[0].length; j++) {
                previousLayerNewWeights[i][j] = currentWeights[i][j] - learningRate * deltas[i] * inputValues[j];
            }
        }
        return previousLayerNewWeights;
    }
}
