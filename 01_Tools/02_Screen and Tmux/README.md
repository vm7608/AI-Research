# **SCREEN AND TMUX IN LINUX**

## **1. Why Screen and Tmux?**

- **Screen** and **Tmux** are terminal multiplexers. They allow you to run multiple terminal sessions inside a single terminal window. You can detach from the session and reattach to it later, which means that you can run a process, detach from it, and then reattach at a later time to see what happened while you were gone.

- **Screen** and **Tmux** are very useful if you are working on a remote machine via SSH, especially if you are working on a slow connection or need to run a process that takes a very long time to complete. It is also useful if you need to leave a process running after you log out of a machine.

## **1. Screen**

- What is a Screen Command? &rarr; Linux Screen is a command-line utility for creating and managing multiple Terminal sessions. It lets you create a virtual shell whose process remains running even after the user disconnects from the server.

- Screen package is pre-installed on most Linux distros. Verify if it is installed on your system using the following command:

```bash
screen --version
```

- If it is not installed, you can install it using the following command:

```bash
sudo apt update
sudo apt install screen
```

- Some of the most common commands used with Screen are as follows:

```bash
# Start a new screen session
screen -S <session_name>

# List all screen sessions
screen -ls

# Reattach to a screen session
screen -r <session_name>

# Detach from a screen session
Ctrl + A + D

# Kill a screen session
screen -X -S <session_name> quit

```

- It's will be a bit uncomfortable when we don't have a scroll bar in the terminal. To scroll, use the following command:

```bash
# Enter copy mode
Ctrl + A + [

# Use up and down buttons or wheel to scroll

# Exit copy mode
Esc
```

- Another way is to modify the configuration file of Screen. Open the configuration file using the following command (I'm not sure it work :3)

```bash
# Open the configuration file
vim ~/.screenrc

# Add the following line to the file
termcapinfo xterm* ti@:te@
termcapinfo xterm* ti@:te@
defscrollback 10000

# Save and exit the file
Esc + :wq
```

## **2. Tmux**

- What is a Tmux Command? &rarr; Tmux is also a terminal multiplexer. It lets you switch easily between several programs in one terminal, detach them (they keep running in the background) and reattach them to a different terminal. And do a lot more.

- Install Tmux using the following command:

```bash
sudo apt-get update
sudo apt-get install tmux
```

- I do not use Tmux much, so I will not write much about it. You can refer to the references below for more information. The following are some of the most common commands used with Tmux:

```bash
# Create a new session
tmux new -s <session_name>

# List all sessions
tmux ls

# Reattach to a session
tmux attach -t <session_name>

# Detach from a session
Ctrl + B + D

# Kill a session
tmux kill-session -t <session_name>

```

- Tmux have much more features than Screen. I will update this section if I switch to Tmux one day =)).

## **3. Conclusion**

- **Tmux** and **Screen** are powerful terminal multiplexers that allow us to work on multiple tasks simultaneously and manage remote sessions easily. Tmux is more advanced in terms of session and window management, and it also provides more customization options. However, it may not be available on all systems and may require some setup. On the other hand, Screen is a more widely available tool that is simpler to use but has fewer features.

- **WARNING:** maybe some of the commands above are not correct, I will update them when I have time. Thank you for reading! ༼ つ ◕_◕ ༽つ

## **4. References**

- [How to use the Linux Screen Command](https://vegastack.com/tutorials/how-to-use-linux-screen-command/)

- [Giới thiệu cơ bản về Tmux](https://viblo.asia/p/gioi-thieu-co-ban-ve-tmux-zoZVRgLEMmg5)

- Thank to Copilot for helping me write this document :3
