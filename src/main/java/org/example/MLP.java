package org.example;

import java.io.IOException;

public class MLP {

    double[] input;
    double[] hidden;
    double outer;
    double[][] wInputHidden;
    double[] wHiddenOuter;


    Double[][] patterns;

    Double[] answer;

    Double[][] test;

    Double[] trainingAnswers;

    public MLP(DataSet dataSet){

        patterns = dataSet.getStudyData();
        answer = dataSet.getStudyAnswers();
        test = dataSet.getTrainData();
        trainingAnswers = dataSet.getTrainAnswers();


        input = new double[patterns[0].length];
        hidden = new double[5];
        wInputHidden = new double[input.length][hidden.length];
        wHiddenOuter = new double[hidden.length];

        initializationWeight();
        study();

        double s = 0;

        for(int p = 0; p < test.length; p++) {
            if(answer[p] == outer)
                s++;
            for (int i = 0; i < input.length; i++) {
                input[i] = test[p][i];
            }

            countOuter();

//            System.out.println(outer);
        }
        System.out.println(s / test.length);
    }

    public void initializationWeight(){
        for (int i = 0; i < wInputHidden.length; i++){
            for (int j = 0; j < wInputHidden[i].length; j++){
                wInputHidden[i][j] = Math.random() * 0.1 + 0.1;
            }
        }
        for (int i = 0; i < wHiddenOuter.length; i++){
            wHiddenOuter[i] = Math.random() * 0.1 + 0.1;
        }
    }

    public void countOuter(){
        for (int i = 0; i < hidden.length; i++){
            hidden[i] = 0;
            for (int j = 0; j < input.length; j++){
                hidden[i] += input[j] * wInputHidden[j][i];
            }
            hidden[i] = (1 / (1 + Math.exp(-hidden[i])));
        }

        outer = 0;

        for (int i = 0; i < hidden.length; i++){
            outer += hidden[i]*wHiddenOuter[i];

        }
        outer = (1 / (1+Math.exp(-outer)));
//        if ((1 / (1+Math.exp(-outer))) > 0.51) outer = 1; else outer = 0;
    }

    public void study(){
        double[] err = new double[hidden.length];
        double gError;
        int count = 0;
        do{
            gError = 0;
            for(int p = 0; p < patterns.length; p++){
                for(int i = 0; i < input.length; i++){
                    input[i] = patterns[p][i];
                }

                countOuter();

                //FIXME исправить формулу вычисления ошибки
                double lError = outer - answer[p];
                gError += (Math.pow(lError, 2))/answer.length;

                for (int i = 0; i < hidden.length; i++){
                    err[i] = lError * wHiddenOuter[i];
                }

                for(int i = 0; i < input.length; i++){
                    for (int j = 0; j < hidden.length; j++){
                        //FIXME исправить формулу
                        wInputHidden[i][j] +=0.1 * err[j] * input[i]; //обуч коэф * знач ошибки скрытом слое * н  а значение входного нейрона
                    }
                }

                for (int i = 0; i < hidden.length; i++) {
                    //FIXME исправить формулу
                    wHiddenOuter[i] += 0.1 * lError * hidden[i]; //корр весов на выходном
                }
            }
            count++;
//            System.out.println(count);
        }
        while (count !=500);
        System.out.println("gError = " + gError);
        System.out.println("Epochs = " + count);
    }

    public static void main(String[] args) throws IOException {
        DataSet dataSet = new DataSet();
//        dataSet.readCSV();
//        dataSet.createTokenWord();
//        dataSet.creteBeforeTokenizedData();
//        dataSet.creteTokenizedData();

        MLP mlp = new MLP(dataSet);
    }
}
