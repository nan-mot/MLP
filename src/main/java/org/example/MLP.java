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

    private Double[] hiddenToOuterLayerWeights;


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


    public Double calculateResult(Double[] input) {
        hiddenLayerResults = calculateHiddenLayerNodeResults(input, inputToHiddenLayerWeights);
        return calculateOuterLayerNodeResult(hiddenLayerResults, hiddenToOuterLayerWeights);
    }


    private void initializationWeight(int inputSize, int hiddenLayerSize, int outerLayerSize) {
        inputToHiddenLayerWeights = new Double[hiddenLayerSize][inputSize];
        for (int i = 0; i < inputToHiddenLayerWeights.length; i++) {
            for (int j = 0; j < inputToHiddenLayerWeights[i].length; j++) {
                inputToHiddenLayerWeights[i][j] = Math.random();
            }
        }
        hiddenToOuterLayerWeights = new Double[hiddenLayerSize];
        for (int i = 0; i < hiddenToOuterLayerWeights.length; i++) {
            hiddenToOuterLayerWeights[i] = Math.random();
        }
    }

    private void study(Double[][] trainingSet, Double[] trainingAnswer) throws IOException {
        Double[] hiddenDelta;
        Double outerDelta;
        Double errorFunction = 0D;
        Double outerLayerResult;
        File errorFile = new File("./gError.txt");
        if (errorFile.exists()) {
            errorFile.delete();
        }
        Path errorFilePath = errorFile.toPath();
        StringBuilder builder = new StringBuilder();
        for (int e = 0; e < epochCount; e++) {
            for (int n = 0; n < trainingSet.length; n++) {
                hiddenLayerResults = calculateHiddenLayerNodeResults(trainingSet[n], inputToHiddenLayerWeights);
                outerLayerResult = calculateOuterLayerNodeResult(hiddenLayerResults, hiddenToOuterLayerWeights);

                errorFunction = calculateErrorFunction(trainingAnswer[n], outerLayerResult);
                builder.append(errorFunction).append(System.lineSeparator());

                outerDelta = outerLevelDeltas(outerLayerResult, trainingAnswer[n]);
                hiddenToOuterLayerWeights = recalculateHiddenOuterLayerWeights(hiddenToOuterLayerWeights, outerDelta, hiddenLayerResults);

                hiddenDelta = hiddenLevelDeltas(outerDelta, hiddenToOuterLayerWeights, hiddenLayerResults);
                inputToHiddenLayerWeights = recalculateLayerWeights(inputToHiddenLayerWeights, hiddenDelta, trainingSet[n]);
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
    private Double[] calculateHiddenLayerNodeResults(Double[] inputValues, Double[][] weights) {
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

    private Double calculateErrorFunction(Double targetValue, Double resultValue) {
        double errorFunction = 0D;
        double toPow = Math.abs(targetValue - resultValue);
        double powResult = Math.pow(toPow, 2D);
        errorFunction += powResult;
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


    private Double outerLevelDeltas(Double outValue, Double answer) {
        return (outValue - answer) * outValue * (1 - outValue);
    }

    private Double[] recalculateHiddenOuterLayerWeights(Double[] currentWeights, Double delta, Double[] outerInput) {
        Double[] newWeights = new Double[currentWeights.length];
        for (int i = 0; i < newWeights.length; i++) {
            newWeights[i] = currentWeights[i] - learningRate * delta * outerInput[i];
        }
        return newWeights;
    }

    private Double[] hiddenLevelDeltas(Double outerDelta, Double[] hiddenToOutWeights, Double[] outputValues) {
        Double[] hiddenLevelDeltas = new Double[outputValues.length];
        for (int i = 0; i < hiddenLevelDeltas.length; i++) {
            hiddenLevelDeltas[i] = outerDelta * hiddenToOutWeights[i] * (outputValues[i] * (1 - outputValues[i]));
        }
        return hiddenLevelDeltas;
    }

    private Double[][] recalculateLayerWeights(Double[][] currentWeights, Double[] deltas, Double[] inputValues) {
        Double[][] previousLayerNewWeights = new Double[currentWeights.length][currentWeights[0].length];
        for (int i = 0; i < deltas.length; i++) {
            for (int j = 0; j < previousLayerNewWeights[0].length; j++) {
                previousLayerNewWeights[i][j] = currentWeights[i][j] - learningRate * deltas[i] * inputValues[j];
            }
        }
        return previousLayerNewWeights;
    }
}
