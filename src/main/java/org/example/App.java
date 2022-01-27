package org.example;

import java.io.IOException;

public class App {
    public static void main(String[] args) throws IOException {
        DataSet dataSet = new DataSet("./irises_dataset_to_study.txt", "./irises_dataset_to_test.txt", 3, ",");
        MLP mlp = new MLP(
                500,
                0.01,
                15,
                3,
                dataSet.getStudyData(),
                dataSet.getStudyAnswers()
        );
        Double[] calculated;
        for (int i = 0; i < dataSet.getTrainData().length; i++) {
            calculated = mlp.calculateResult(dataSet.getTrainData()[i]);
            System.out.println(
                    "Calculated: " + dataSet.getAnswerString(calculated) +
                            "  Real: " + dataSet.getAnswerString(dataSet.getTrainAnswers()[i]));
        }
    }
}
