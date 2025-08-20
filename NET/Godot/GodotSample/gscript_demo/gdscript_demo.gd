extends Node

@onready var datafeel_sdk_node: Node = get_node("DatafeelSDK")
@onready var connected_container: Control = get_node("ConnectedContainer")
@onready var disconnected_container: Control = get_node("DisconnectedContainer")

@onready var color_picker: ColorPicker = get_node("ConnectedContainer/ColorPicker")
@onready var vibrate_button: Button = get_node("ConnectedContainer/VibrateButton")
@onready var vibrate_stop_button: Button = get_node("ConnectedContainer/StopVibrateButton")
@onready var intensity_slider: HSlider = get_node("ConnectedContainer/IntensitySlider")
@onready var frequency_slider: HSlider = get_node("ConnectedContainer/FrequencySlider")
@onready var temperature_c_slider: HSlider = get_node("ConnectedContainer/TemperatureCSlider")

@onready var connect_button: Button = get_node("DisconnectedContainer/ConnectButton")
@onready var disconnect_button: Button = get_node("ConnectedContainer/DisconnectButton")

var device_dot_index = 0


func _ready() -> void:
	print("Start Datafeel GDScript Demo")
	_connect_ui_signals()


func _process(_delta: float) -> void:
	_update_connection_state()


func _update_connection_state() -> void:
	connected_container.visible = datafeel_sdk_node.Connected
	disconnected_container.visible = not datafeel_sdk_node.Connected
	disconnect_button.disabled = not datafeel_sdk_node.Connected


func _connect_ui_signals() -> void:
	color_picker.color_changed.connect(_on_color_changed)
	vibrate_button.pressed.connect(_on_vibrate_pressed)
	vibrate_stop_button.pressed.connect(_on_stop_vibrate_pressed)
	intensity_slider.value_changed.connect(_on_intensity_changed)
	frequency_slider.value_changed.connect(_on_frequency_changed)
	temperature_c_slider.value_changed.connect(_on_temperature_c_changed)
	disconnect_button.pressed.connect(_on_disconnect_pressed)
	connect_button.pressed.connect(_on_connect_pressed)


func _on_connect_pressed() -> void:
	_connect_device()


func _on_disconnect_pressed() -> void:
	_disconnect_device()


func _connect_device() -> void:
	datafeel_sdk_node.Connect()
	print("Start Connect from GDScript")
	connect_button.disabled = true
	await datafeel_sdk_node.ConnectComplete
	connect_button.disabled = false
	print("Connection Established: %s" % datafeel_sdk_node.Connected)


func _disconnect_device() -> void:
	datafeel_sdk_node.Disconnect()
	print("Disconnect DataFeel device.")


func _on_color_changed(color: Color) -> void:
	datafeel_sdk_node.SetGlobalColor(device_dot_index, color)


func _on_vibrate_pressed() -> void:
	var seconds = 1.0
	var intensity = 0.5
	var frequency = 128.0
	vibrate_button.disabled = true
	datafeel_sdk_node.VibrateForSeconds(device_dot_index, seconds, intensity, frequency)
	await get_tree().create_timer(1.0).timeout
	vibrate_button.disabled = false


func _on_stop_vibrate_pressed() -> void:
	datafeel_sdk_node.StopVibration(device_dot_index)


func _on_intensity_changed(_intensity: float) -> void:
	datafeel_sdk_node.SetVibration(device_dot_index, intensity_slider.value, frequency_slider.value)


func _on_frequency_changed(_frequency: float) -> void:
	datafeel_sdk_node.SetVibration(device_dot_index, intensity_slider.value, frequency_slider.value)


func _on_temperature_c_changed(value: float) -> void:
	datafeel_sdk_node.SetTemperatureC(device_dot_index, value)
