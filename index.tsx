import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, BackHandler, Alert } from 'react-native';
import { StatusBar } from 'expo-status-bar';

export default function App() {
  const [task, setTask] = useState<string>('');
  const [duration, setDuration] = useState<string>('');
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [pauseCount, setPauseCount] = useState<number>(0);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startTimer = () => {
    if (!task || !duration || isNaN(Number(duration)) || Number(duration) <= 0) {
      Alert.alert('Error', 'Enter a task and a valid duration.');
      return;
    }
    const seconds = parseInt(duration) * 60;
    setTimeLeft(seconds);
    setIsRunning(true);
    setPauseCount(0);
    setIsPaused(false);
  };

  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;
    if (isRunning && !isPaused && timeLeft > 0) {
      interval = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);
    } else if (timeLeft === 0 && isRunning) {
      setIsRunning(false);
      Alert.alert('Done!', 'Time’s up!');
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning, isPaused, timeLeft]);

  const pauseTimer = () => {
    if (pauseCount < 3) {
      setIsPaused(true);
      setPauseCount(pauseCount + 1);
    } else {
      Alert.alert('No More Pauses', 'You’ve used all 3 pauses!');
    }
  };

  const resumeTimer = () => {
    setIsPaused(false);
  };

  useEffect(() => {
    const backHandler = BackHandler.addEventListener('hardwareBackPress', () => {
      if (isRunning) {
        Alert.alert('Stay Here', 'You can’t leave during the timer!');
        return true;
      }
      return false;
    });
    return () => backHandler.remove();
  }, [isRunning]);

  return (
    <View style={styles.container}>
      {!isRunning ? (
        <View>
          <TextInput
            style={styles.input}
            placeholder="Task name"
            placeholderTextColor="#888"
            value={task}
            onChangeText={setTask}
          />
          <TextInput
            style={styles.input}
            placeholder="Minutes"
            placeholderTextColor="#888"
            value={duration}
            onChangeText={setDuration}
            keyboardType="numeric"
          />
          <Button title="Start" onPress={startTimer} color="#444" />
        </View>
      ) : (
        <View>
          <Text style={styles.task}>{task}</Text>
          <Text style={styles.timer}>{formatTime(timeLeft)}</Text>
          <Text style={styles.pauses}>Pauses left: {3 - pauseCount}</Text>
          {isPaused ? (
            <Button title="Resume" onPress={resumeTimer} color="#444" />
          ) : (
            <Button title="Pause" onPress={pauseTimer} disabled={pauseCount >= 3} color="#444" />
          )}
        </View>
      )}
      <StatusBar style="light" backgroundColor="#1a1a1a" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  task: { color: '#fff', fontSize: 24, marginBottom: 20 },
  timer: { color: '#fff', fontSize: 48, marginBottom: 20 },
  pauses: { color: '#888', fontSize: 16, marginBottom: 20 },
  input: {
    backgroundColor: '#222',
    color: '#fff',
    padding: 10,
    marginVertical: 10,
    width: 200,
    borderRadius: 5,
    borderColor: '#444',
    borderWidth: 1,
  },
});