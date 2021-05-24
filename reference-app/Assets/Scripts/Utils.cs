using System;
using System.Globalization;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;

public class ScreenConsoleLogger
{
    private readonly Text _consoleText;

    public ScreenConsoleLogger(Text consoleText)
    {
        _consoleText = consoleText;
    }

    public void Info(string message)
    {
        var d = DateTime.Now.ToString("MM/dd/yyyy HH:mm:ss");
        _consoleText.text += $"[INFO] [{d}] {message}\n";
    }

    public void Error(string message)
    {
        var d = DateTime.Now.ToString("MM/dd/yyyy HH:mm:ss");
        _consoleText.text += $"[ERROR] [{d}] <<{message}>>\n";
    }

    public void Measure(string identifier, string mark, DateTime baseTime)
    {
        var n = DateTime.Now;
        var d = n.ToString("MM/dd/yyyy HH:mm:ss");
        var duration = (n - baseTime).TotalMilliseconds.ToString(CultureInfo.InvariantCulture);
        _consoleText.text += $"[Measure] [{n}] {identifier}, {mark}, {duration}ms\n";
    }
}

public static class Utils
{
    public static void BindButtonClick(string buttonName, UnityAction handler)
    {
        GameObject
            .Find(buttonName)
            .GetComponent<Button>()
            .onClick.AddListener(handler);
    }
}