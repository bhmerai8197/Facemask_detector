String dev_name = "Arduino#" ;
String message;
int out_sensor = 0;
int switch_pin = 14;

void setup() {

  //Initialise communication with system over serial
  Serial.begin(9600);
  
  pinMode(A14,INPUT);
  pinMode(A15,INPUT);
  pinMode(13,OUTPUT);
  pinMode(switch_pin,INPUT_PULLUP);

  digitalWrite(13,HIGH);

  //Wait for bootup command from system
  
  while(!Serial.available()) ;
    
  message = Serial.readStringUntil('#');// read the incoming data as string

  Serial.println("Device Booted Successfully");

  digitalWrite(13,LOW);
  
}


//Function if command to open gates is received
int GateOpen()
{
  
  unsigned long int timeout = millis();
  int people_count = 0;
  int timeout_delay = 6000;

  //Send 0 if no person walks in within 6 seconds or return walk in count
  while( (analogRead(A14) < 550) && (analogRead(A14) > 480) && ((millis() - timeout) < timeout_delay )) ;

  delay(400);
  people_count = abs((512 -analogRead(A14)))/125;  

  return people_count;    
  
}

void loop() {

  int people_count = 0;

  //Return count of people walking out
  if(digitalRead(switch_pin) == LOW)
  {
      Serial.println("-1");
      delay(100);
  }
  
  if(Serial.available())
  {
   
   message = Serial.readStringUntil('#');// read the incoming data as string

   if (message == "Gates")
     {
      people_count = GateOpen();
      Serial.println(people_count);
      
      digitalWrite(13,HIGH);
     
     }

  if (message == "Manual")
    {
      Serial.println("Manual Override Detected");
      digitalWrite(13,LOW);
     
    }

  if(message == "Manual_Over")
    {
      Serial.println("Manual Override Removed");
       digitalWrite(13,HIGH);
     
    }
  message = "NULL";
  
  }

  delay(80);

}
