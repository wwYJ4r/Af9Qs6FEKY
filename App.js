/*
@name: Gabriel Jucá
@date: 21st Nov 2021
App.js

Front-end of a Travel App application that allows the user to share expenses of a trip with their friends.
This code was done as an integrated assignment for the Higher Diploma in Science in Computing (CCT Dublin).

 */
import { StatusBar } from "expo-status-bar";
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

import LoginPage from "./LoginPage"; //Welcome screen of the app (login)
import MenuPage from "./MenuPage"; //Menu screen for navigation between different features of the app
import ExpensesPage from "./ExpensesPage"; //Expenses screen to see list of expenses
import PostPage from "./PostPage"; //Post screen to post new expenses
import FinishPage from "./FinishPage"; //Finish screen to get the final balance of the trip

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      {/* Setting the screens and their titles */}
      <Stack.Navigator>
        <Stack.Screen
          name="Login"
          component={LoginPage}
          options={{ title: "Login" }}
        />
        <Stack.Screen
          name="Menu"
          component={MenuPage}
          options={{ title: "Menu" }}
        />
        <Stack.Screen
          name="Expenses"
          component={ExpensesPage}
          options={{ title: "See expenses" }}
        />
        <Stack.Screen
          name="Post"
          component={PostPage}
          options={{ title: "Post new expense" }}
        />
        <Stack.Screen
          name="Finish"
          component={FinishPage}
          options={{ title: "Finish a trip" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});
