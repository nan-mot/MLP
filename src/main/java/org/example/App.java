package org.example;

import java.io.IOException;

public class App {
    public static void main(String[] args) throws IOException {
        DataSet dataSet = new DataSet();
        MLP mlp = new MLP(
                500,
                0.4,
                100,
                1,
                dataSet.getStudyData(),
                dataSet.getStudyAnswers()
        );
        for (int i = 0; i < dataSet.getTrainData().length; i++) {
            System.out.println(
                    "Calculated: " + mlp.calculateResult(dataSet.getTrainData()[i])[0] +
                            "  Real: " + dataSet.getTrainAnswers()[i]);
        }
    }
}
