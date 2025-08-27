int pin_Out_S0 = 0;
int pin_Out_S1 = 1;
int pin_Out_S2 = 2;
int pin_In_Mux1 = A3;
int Mux1_State[8] = { 0 };
unsigned long previousMillis = 0;        // will store last time LED was updated

// constants won't change :
const long interval = 2000;           // interval at which to blink (milliseconds)

void setup() {
  pinMode(pin_Out_S0, OUTPUT);
  pinMode(pin_Out_S1, OUTPUT);
  pinMode(pin_Out_S2, OUTPUT);
  //pinMode(pin_In_Mux1, INPUT);
  Serial.begin(115200);
  
}

void loop() {
  updateMux1();
  String line = "";
  for (int i = 0; i < 8; i++) {
    line += String(Mux1_State[i]);
    if (i < 7) {
      line += ",";
    }
  }
  Serial.println(line);
}


void updateMux1() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    for (int i = 0; i < 8; i++) {
      digitalWrite(pin_Out_S0, HIGH && (i & B00000001));
      digitalWrite(pin_Out_S1, HIGH && (i & B00000010));
      digitalWrite(pin_Out_S2, HIGH && (i & B00000100));
      Mux1_State[i] = analogRead(pin_In_Mux1);
    }
  }
}
