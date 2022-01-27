package org.example;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class DataSet {

    private Double[][] studyData;
    private Double[][] studyAnswers;
    private Double[][] data;
    private Double[][] answers;

    private Map<String, Double[]> answersMatchingMap = new HashMap<>();
    private Map<Integer, String> answerByOrderNumber = new HashMap<>();
    private int answerCounter = 0;
    private int totalAnswers;


    public DataSet(String studyDatasetFilePath, String testDatasetFilePath, int totalAnswers, String delimiter) throws IOException {
        this.totalAnswers = totalAnswers;

        Pair<Double[][], Double[][]> parsedData;
        parsedData = parseDataset(studyDatasetFilePath, delimiter);
        studyData = parsedData.first;
        studyAnswers = parsedData.second;

        parsedData = parseDataset(testDatasetFilePath, delimiter);
        data = parsedData.first;
        answers = parsedData.second;
        System.out.println("Parsed");
    }

    private Pair<Double[][], Double[][]> parseDataset(String filePath, String delimiter) throws IOException {
        File file = new File(filePath);
        List<String> lines = Files.readAllLines(file.toPath());
        return splitData(lines, delimiter);
    }

    private Pair<Double[][], Double[][]> splitData(List<String> lines, String delimiter) {
        Double[][] data = new Double[lines.size()][lines.get(0).split(delimiter).length - 1];
        Double[][] answers = new Double[lines.size()][totalAnswers];
        String[] split;
        for (int i = 0; i < lines.size(); i++) {
            split = lines.get(i).split(",");
            for (int j = 0; j < split.length; j++) {
                if (j == split.length - 1) {
                    answers[i] = parseAnswer(split[j]);
                } else {
                    data[i][j] = parseDouble(split[j]);
                }
            }
        }
        return new Pair<>(data, answers);
    }

    private Double parseDouble(String s) {
        try {
            return Double.parseDouble(s);
        } catch (Exception e) {
            System.out.println("Incorrect value: " + s);
            return 0.0;
        }
    }

    public Double[] parseAnswer(String answer) {
        if (answersMatchingMap.containsKey(answer)) {
            return answersMatchingMap.get(answer);
        } else {
            Double[] newAnswer = new Double[totalAnswers];
            for (int i = 0; i < totalAnswers; i++) {
                newAnswer[i] = 0D;
            }
            newAnswer[answerCounter] = 1D;
            answersMatchingMap.put(answer, newAnswer);
            answerByOrderNumber.put(answerCounter, answer);
            answerCounter++;
            return newAnswer;
        }
    }

    public String getAnswerString(Double[] answerVector) {
        Double maxValue = Double.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < answerVector.length; i++) {
            if (answerVector[i] > maxValue) {
                maxValue = answerVector[i];
                index = i;
            }
        }
        return answerByOrderNumber.get(index);
    }

    public Double[][] getStudyAnswers() {
        return studyAnswers;
    }

    public Double[][] getStudyData() {
        return studyData;
    }

    public Double[][] getTrainData() {
        return data;
    }

    public Double[][] getTrainAnswers() {
        return answers;
    }

}
