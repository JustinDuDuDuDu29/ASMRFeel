using Datafeel;
using Datafeel.NET.Serial;

var manager = new DotManagerConfiguration()
    .AddDot<Dot_63x_xxx>(1)
    .AddDot<Dot_63x_xxx>(2)
    .AddDot<Dot_63x_xxx>(3)
    .AddDot<Dot_63x_xxx>(4)
    .CreateDotManager();

try
{
    using var cts = new CancellationTokenSource(1000);
    var serialClient = new DatafeelModbusClientConfiguration()
        .UseWindowsSerialPortTransceiver()
        //.UseSerialPort("COM3") // Uncomment this line to specify the serial port by name
        .CreateClient();
    var result = await manager.Start(serialClient, cts.Token);
    if (result)
    {
        Console.WriteLine("Started");
    }
    else
    {
        Console.WriteLine("Failed to start");
    }
}
catch (Exception e)
{
    Console.WriteLine(e.Message);
}

while (true)
{
    var delay = Task.Delay(2000);
    foreach (var d in manager.Dots)
    {
        d.VibrationMode = VibrationModes.Library;
        //There can be up to 8 waveforms in the sequence
        d.VibrationSequence[0].Waveforms = VibrationWaveforms.StrongBuzzP100;
        d.VibrationSequence[1].RestDuration = 500; // Milliseconds
        d.VibrationSequence[2].Waveforms = VibrationWaveforms.TransitionHum1P100;
        d.VibrationSequence[3].Waveforms = VibrationWaveforms.TransitionRampDownMediumSharp2P50ToP0;
        d.VibrationSequence[4].Waveforms = VibrationWaveforms.TransitionRampUpShortSharp2P0ToP50;
        d.VibrationSequence[5].RestDuration = 500; // Milliseconds
        d.VibrationSequence[6].Waveforms = VibrationWaveforms.TransitionRampUpShortSmooth2P0ToP100;
        d.VibrationSequence[7].Waveforms = VibrationWaveforms.StrongClick1P100;

        // Use VibrationWaveforms.EndSequence at any index to truncate the sequence instead of having the full 8 waveforms

        // Set this to true, then call Write() to flush the values and start the sequence
        d.VibrationGo = true;

        try
        {
            await d.Write();
        }
        catch (Exception e)
        {
            Console.WriteLine(e.Message);
        }
    }
    await delay;
}
