namespace Datafeel;

using Godot;
using System;
using System.Threading;
using Datafeel.NET.BLE;
using Datafeel.NET.Serial;
using System.Threading.Tasks;
using System.Collections.Generic;

public partial class DatafeelGodotSdk : Node
{
	private const int MAX_DOTS = 4;
	private const string PORT_NAME = "";

	private DotManager _dotManager;

	public bool Connected => _dotManager?.IsRunning == true;

	public ManagedDot[] Dots {get; private set; }

	[Signal] public delegate void ConnectCompleteEventHandler(bool success);
	[Signal] public delegate void DeviceReadCompleteEventHandler(bool success);

	public override void _Ready()
	{
		Initialize();
	}

	public void Initialize()
	{
		GD.Print("Initializing Datafeel Godot SDK");
		_dotManager = new DotManagerConfiguration().CreateDotManager();
		Dots = new ManagedDot[MAX_DOTS];
		for (int i = 0; i < MAX_DOTS; i++)
		{
			byte address = (byte)(Convert.ToByte(i) + Convert.ToByte(1));
			var dotFound = _dotManager.AddDot(address);
			if (!dotFound)
			{
				continue;
			}
			Dots[i] = _dotManager.FindDot(address);
			Dots[i].LedMode = LedModes.GlobalManual;
			Dots[i].VibrationMode = VibrationModes.Manual;
		}
	}
	
	public void Connect()
	{
		_ = ConnectAsync();
	}

	public void Disconnect()
	{
		_dotManager?.Stop();
	}

	public async Task<bool> ConnectAsync()
	{
		if (_dotManager is null)
		{
			Initialize();
		}
		GD.Print("Starting Datafeel Device Connection.");
		using var cts = new CancellationTokenSource(10000);
		try
		{
			DatafeelModbusClient serialClient;
			if (OS.GetName() == "Windows")
			{
				serialClient = new DatafeelModbusClientConfiguration()
					.UseWindowsSerialPortTransceiver()
					.CreateClient();
			}
			else
			{
				serialClient = new DatafeelModbusClientConfiguration()
					.UseSerialPort(PORT_NAME)
					.CreateClient();
			}
			var bleClient = new DatafeelModbusClientConfiguration()
				.UseNetBleTransceiver()
				.CreateClient();
			var clients = new List<DatafeelModbusClient> { serialClient, bleClient };
			await _dotManager.Start(clients, cts.Token);
			GD.Print($"Connected: {Connected}");
			if (Connected)
			{
				GD.Print("Datafeel Device Connection Established!");
			}
			else
			{
				GD.Print("Datafeel Connection was not established.");
			}
			EmitSignal(SignalName.ConnectComplete, Connected);
			return Connected;
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
		EmitSignal(SignalName.ConnectComplete, Connected);
		return Connected;
	}

	public void SetLedMode(int index, LedModes mode)
	{
		if (index < 0 || index > Dots.Length)
		{
			return;
		}
		Dots[index].LedMode = mode;
		try
		{
			Dots[index].Write(default);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
	}
	
	public void SetTemperatureC(int index, float temperatureC)
	{
		if (index < 0 || index > Dots.Length)
		{
			return;
		}
		Dots[index].TargetSkinTemperature = temperatureC;
		Dots[index].ThermalMode = ThermalModes.TemperatureTarget;
		try
		{
			Dots[index].Write(default);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
	}

	public void SetGlobalColor(int index, Color color)
	{
		if (index < 0 || index > Dots.Length)
		{
			return;
		}
		Dots[index].GlobalLed.Red = (byte) color.R8;
		Dots[index].GlobalLed.Green = (byte) color.G8;
		Dots[index].GlobalLed.Blue = (byte) color.B8;
		try
		{
			Dots[index].Write(default);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
	}

	public void SetVibrationMode(int index, VibrationModes mode)
	{
		if (index < 0 || index > Dots.Length)
		{
			return;
		}
		Dots[index].VibrationMode = mode;
		try
		{
			Dots[index].Write(default);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
	}

	public async Task ReadAsync(int index)
	{
		try
		{
			await Dots[index].Read();
			EmitSignal(SignalName.DeviceReadComplete, true);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
			EmitSignal(SignalName.DeviceReadComplete, false);
		}
	}

	public async void SetVibration(int index, float intensity=0.5f, float frequency=128f)
	{
		Dots[index].VibrationMode = VibrationModes.Manual;
		Dots[index].VibrationIntensity = intensity;
		Dots[index].VibrationFrequency = frequency;
		try
		{
			await Dots[index].Write(default);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
	}

	public void StopVibration(int index)
	{
		SetVibration(index, 0f, 0f);
	}

	public void VibrateForSeconds(int index, float seconds=0.5f, float intensity=0.5f, float frequency=128f)
	{
		_ = VibrateForSecondsAsync(index, seconds, intensity, frequency);
	}

	public async Task VibrateForSecondsAsync(int index, float seconds=1f, float intensity=0.5f, float frequency=128f)
	{
		Dots[index].VibrationMode = VibrationModes.Manual;
		Dots[index].VibrationIntensity = intensity;
		Dots[index].VibrationFrequency = frequency;
		
		try
		{
			await Dots[index].Write(default);
			await Task.Delay(Mathf.FloorToInt(seconds * 1000f));
			Dots[index].VibrationIntensity = 0f;
			Dots[index].VibrationFrequency = 0f;
			await Dots[index].Write(default);
		}
		catch (Exception e)
		{
			GD.PushError(e.Message);
		}
	}
}
