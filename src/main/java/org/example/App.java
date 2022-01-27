package org.example;

import java.io.IOException;

public class App {
    public static void main(String[] args) throws IOException {
        DataSet dataSet = new DataSet();
        MLP mlp = new MLP(
                500,
                0.2,
                8,
                1,
                dataSet.getStudyData(),
                dataSet.getStudyAnswers()
        );
        Double[] calculated;
        for (int i = 0; i < dataSet.getTrainData().length; i++) {
            calculated = mlp.calculateResult(dataSet.getTrainData()[i]);
            System.out.println(
                    "Calculated: " + Math.round(calculated[0]) +
                            "  Real: " + dataSet.getTrainAnswers()[i]);
        }
    }
}
