package org.example;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;


public class DataSet {
    private String datasetFilePath = "C:\\study\\vtb\\MLP\\processed.cleveland.data";
    private Double[][] studyData;
    private Double[] studyAnswers;
    private Double[][] data;
    private Double[] answers;

    public DataSet() throws IOException {
        readCSV();
    }

    private void splitData(int i, String line, Double[][] data, Double[] answers) {
        String[] split = line.split(",");
        for (int j = 0; j < split.length; j++) {
                if (j == split.length - 1) {
                    answers[i] = parseDouble(split[j]);
                } else {
                    data[i][j] = parseDouble(split[j]);
                }
        }
    }

    private Double parseDouble(String s) {
        try {
            return Double.parseDouble(s);
        } catch (Exception e) {
            System.out.println("Incorrect value: " + s);
            return 0.0;
        }
    }

    public void readCSV() throws IOException {
        File file = new File(datasetFilePath);
        List<String> lines = Files.readAllLines(file.toPath());
        int toStudy = 180;
        studyData = new Double[toStudy][13];
        studyAnswers = new Double[toStudy];
        data = new Double[lines.size() - toStudy][13];
        answers = new Double[lines.size() - toStudy];
        for (int i = 0; i < lines.size(); i++) {
            if (i < toStudy) {
                splitData(i, lines.get(i), studyData, studyAnswers);
            } else {
                splitData(i - toStudy, lines.get(i), data, answers);
            }
        }

    }

    public Double[] getStudyAnswers() {
        return studyAnswers;
    }

    public Double[][] getStudyData() {
        return studyData;
    }

    public Double[][] getTrainData() {
        return data;
    }

    public Double[] getTrainAnswers() {
        return answers;
    }

}
