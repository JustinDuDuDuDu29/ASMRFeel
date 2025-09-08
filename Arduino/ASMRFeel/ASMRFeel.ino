int pin_Out_S0 = 0;
int pin_Out_S1 = 1;
int pin_Out_S2 = 2;
int pin_In_Mux1 = A3;
int pin_In_Mux2 = A4;
int Mux1_State[16] = { 0 };
int btn = 10;
int buttonState;
int lastButtonState = LOW;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;
unsigned long previousMillis = 0;
int LED = 13;
const long interval = 150;
bool hot = true;
void setup() {
  pinMode(pin_Out_S0, OUTPUT);
  pinMode(pin_Out_S1, OUTPUT);
  pinMode(pin_Out_S2, OUTPUT);
  pinMode(btn, INPUT);
  //pinMode(pin_In_Mux1, INPUT);

  Serial.begin(115200);
  Serial.flush();
  Serial.clear();
}

void loop() {
  unsigned long currentMillis = millis();

  int reading = digitalRead(btn);
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      if (buttonState == HIGH) {
        hot = !hot;
      }
    }
  }

  digitalWrite(LED, hot);

  lastButtonState = reading;

  if (currentMillis - previousMillis >= interval) {
    updateMux();
    String line = "";
    // for (int i = 0; i < 8; i++) {
    //   line += String(Mux1_State[i]);
    //   if (i < 7) {
    //     line += ",";
    //   }
    // }
    if (hot) {
      line += "1;";
    } else {
      line += "0;";
    }

    for (int i = 0; i < 16; i++) {
      line += String(Mux1_State[i]);
      if (i == 7) {
        line += ";";
      } else if (i < 15) {
        line += ",";
      }
    }
    Serial.println(line);

    previousMillis = currentMillis;
  }
}


void updateMux() {

  for (int i = 0; i < 8; i++) {
    digitalWrite(pin_Out_S0, HIGH && (i & B00000001));
    digitalWrite(pin_Out_S1, HIGH && (i & B00000010));
    digitalWrite(pin_Out_S2, HIGH && (i & B00000100));
    delayMicroseconds(6);           // tune 5–50 µs as needed
    (void)analogRead(pin_In_Mux1);  // throw away first read (S/H settling)
    (void)analogRead(pin_In_Mux2);  // throw away first read (S/H settling)
    Mux1_State[i] = analogRead(pin_In_Mux1);
    Mux1_State[i + 8] = analogRead(pin_In_Mux2);
  }
}