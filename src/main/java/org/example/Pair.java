package org.example;

public class Pair<I, J> {

    public Pair(I first, J second) {
        this.first = first;
        this.second = second;
    }

    final I first;
    final J second;
}
