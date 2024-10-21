import sys
import os
import socket
import pyaudio
import wave

import speech_recognition as sr
import json
# import psutil
import traceback
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QSplitter, QDialog, 
                             QLineEdit, QGridLayout, QSizePolicy)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QObject
from PyQt6.QtGui import QFont, QColor, QTextCursor

from PyQt6.QtWidgets import QComboBox, QSpacerItem, QSizePolicy
import random

# set server ip, and post number here
server_ip = "localhost"
server_port = 8086

class GlobalExceptionHandler(QObject):
    exception_caught = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"Uncaught exception:\n{error_msg}")
        self.exception_caught.emit(error_msg)

def install_global_exception_handler(app):
    exception_handler = GlobalExceptionHandler()
    sys.excepthook = exception_handler.handle_exception
    return exception_handler


class AudioRecorderThread(QThread):
    finished = pyqtSignal()

    def __init__(self, audio, frames):
        super().__init__()
        self.audio = audio
        self.frames = frames
        self.is_recording = True

    def run(self):
        stream = self.audio.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=44100,
                                 input=True,
                                 frames_per_buffer=1024)
        
        while self.is_recording:
            data = stream.read(1024)
            self.frames.append(data)
        
        stream.stop_stream()
        stream.close()
        self.finished.emit()

class AudioRecorderWidget(QWidget):
    transcription_ready = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.filename = "recorded_audio.wav"
        self.recorder_thread = None
        self.recording_time = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.setFixedSize(500, 200)

    def initUI(self):
        layout = QVBoxLayout()

        self.timer_label = QLabel('00:00', self)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setFont(QFont('Arial', 36, QFont.Weight.Bold))
        self.timer_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(self.timer_label)

        
        self.record_button = QPushButton('Start Recording', self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.record_button.setFont(QFont('Arial', 18, QFont.Weight.Bold))
        self.record_button.setFixedHeight(80)  # Increase button height
        layout.addWidget(self.record_button)


        self.status_label = QLabel('', self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont('Arial', 14))
        self.status_label.setStyleSheet("color: #34495e;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.setText('Stop Recording')
        self.status_label.setText('Recording...')
        self.status_label.setStyleSheet("color: #e74c3c;")

        self.frames = []
        self.recorder_thread = AudioRecorderThread(self.audio, self.frames)
        self.recorder_thread.finished.connect(self.on_recording_finished)
        self.recorder_thread.start()

        self.recording_time = 0
        self.timer.start(1000)

    def stop_recording(self):
        if self.recorder_thread:
            self.recorder_thread.is_recording = False
        self.timer.stop()
        self.timer_label.setText(f'00:00')
        self.status_label.setText('')
      

    def on_recording_finished(self):
        self.is_recording = False
        self.record_button.setText('Start Recording')
        # self.status_label.setText('Waiting for your turn ...')
        self.status_label.setStyleSheet("color: #27ae60;")

        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.transcribe_audio()

    def transcribe_audio(self):
        # self.status_label.setText('Transcribing audio...')
        self.status_label.setStyleSheet("color: #f39c12;")
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.filename) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            if not text.strip():  # If the transcription is empty or just whitespace
                text = ":empty:"
            # self.status_label.setText('Transcription completed')
            self.status_label.setStyleSheet("color: #27ae60;")
            self.transcription_ready.emit(text)
        except sr.UnknownValueError:
            # self.status_label.setText('Speech Recognition could not understand audio')
            self.status_label.setStyleSheet("color: #e74c3c;")
            self.transcription_ready.emit(":empty:")
        except sr.RequestError as e:
            # self.status_label.setText('Could not request results from Speech Recognition service')
            self.status_label.setStyleSheet("color: #e74c3c;")
            self.transcription_ready.emit(":empty:")
        finally:
            if os.path.exists(self.filename):
                os.remove(self.filename)
                print(f"Deleted recording: {self.filename}")

    def update_timer(self):
        self.recording_time += 1
        minutes = self.recording_time // 60
        seconds = self.recording_time % 60
        self.timer_label.setText(f'{minutes:02d}:{seconds:02d}')

class StoryboardWidget(QWidget):
    word_selected_signal = pyqtSignal(str)

    def __init__(self, words_with_weights, parent=None):
        super().__init__(parent)
        self.words_with_weights = words_with_weights
        self.colors = self.generate_colors(len(words_with_weights))
        self.crossed_words = set()
        self.initUI()

    def generate_colors(self, num_colors):
        # Carefully curated color palette based on color theory
        # These colors are harmonious and provide good contrast with white text
        color_palette = [
            "#3498DB",  # Blue
            "#E74C3C",  # Red
            "#2ECC71",  # Green
            "#F39C12",  # Orange
            "#9B59B6",  # Purple
            "#1ABC9C",  # Turquoise
            "#D35400",  # Pumpkin
            "#34495E",  # Navy Blue
            "#27AE60",  # Emerald
            "#E67E22",  # Carrot
            "#8E44AD",  # Wisteria
            "#16A085"   # Green Sea
        ]

        # Generate the requested number of colors
        result = []
        for i in range(num_colors):
            color_hex = color_palette[i % len(color_palette)]
            result.append(QColor(color_hex))

        return result

    def initUI(self):
        grid_layout = QGridLayout()
        self.buttons = []

        for i, (word, weight) in enumerate(self.words_with_weights.items()):
            button = QPushButton(self)
            button.setText(f"{word}\n{weight}")
            button.setFont(QFont("Arial", 18))
            button.setStyleSheet(self.get_button_style(self.colors[i]))
            button.clicked.connect(lambda checked, w=word: self.word_selected(w))
            self.buttons.append(button)
            row, col = divmod(i, 3)
            grid_layout.addWidget(button, row, col)

            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        grid_layout.setSpacing(5)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(grid_layout)
        self.setFixedSize(500, 500)

    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color.name()};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {color.lighter(150).name()};
            }}
            QPushButton:pressed {{
                background-color: {color.darker(150).name()};
            }}
        """

    def word_selected(self, word):
        if word not in self.crossed_words:
            self.crossed_words.add(word)
            print(f"Selected word: {word}")
            self.update_button_styles()
            self.word_selected_signal.emit(word)

    def update_button_styles(self):
        for button in self.buttons:
            word = button.text().split('\n')[0]
            if word in self.crossed_words:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: gray;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        text-decoration: line-through;
                        padding: 10px;
                        text-align: center;
                    }
                """)
            else:
                button.setStyleSheet(self.get_button_style(self.colors[list(self.words_with_weights.keys()).index(word)]))

class NameInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Participant Information")
        self.setFixedSize(1600, 900)
        self.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #2c3e50;
            }
        """)
 
        name_input_style = "font-size: 24px; height: 40px;"
        label_style = "font-size: 24px; color: #2c3e50;"
        combo_box_style = "font-size: 20px; padding: 10px; height: 40px;"
        push_button_style = "background-color: #3498db; color: white; border: none; padding: 15px 30px; font-size: 24px; border-radius: 5px;"
        
        layout = QVBoxLayout()

        # Name input field
        name_label = QLabel("Enter your name:")
        name_label.setStyleSheet(label_style)
        self.name_input = QLineEdit()
        self.name_input.setFixedWidth(600)
        self.name_input.setStyleSheet(name_input_style)

        # Participant ID ComboBox
        participant_label = QLabel("Select your Participant ID:")
        participant_label.setStyleSheet(label_style)
        self.participant_combo = QComboBox()
        self.participant_combo.setFixedWidth(600)
        self.participant_combo.setStyleSheet(combo_box_style)
        self.participant_combo.addItems([f"pid_{i}" for i in range(71)])

        # Spacer to add vertical gap between name input and combo box
        vertical_spacer_1 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        # Story Preference ComboBox
        preference_label = QLabel("Select your story preference:")
        preference_label.setStyleSheet(label_style)
        self.preference_combo = QComboBox()
        self.preference_combo.setFixedWidth(600)
        self.preference_combo.setStyleSheet(combo_box_style)

        # Theme options as provided
        theme_list = ["adventure", "fairy tale", "mystery", "teamwork" ]
        random_themes = random.sample(theme_list, 4)
        self.preference_combo.addItems(random_themes)

        # Spacer to add vertical gap before narrative
        vertical_spacer_2 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        # Narrative writing level
        narrative_label = QLabel("Write a narrative using the word 'Brave' on the theme 'Journey':")
        narrative_label.setStyleSheet(label_style)
        self.narrative_input = QTextEdit()
        self.narrative_input.setFixedSize(800, 300)
        self.narrative_input.setStyleSheet("font-size: 20px; padding: 10px;")

        # Spacer to add vertical gap before the submit button
        vertical_spacer_3 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.setFixedWidth(300)
        self.submit_button.setStyleSheet(push_button_style)

        # Add widgets to the layout
        layout.addWidget(name_label)
        layout.addWidget(self.name_input)

        layout.addWidget(participant_label)
        layout.addWidget(self.participant_combo)

        layout.addItem(vertical_spacer_1)  # Add spacer between name input and combo box

        layout.addWidget(preference_label)
        layout.addWidget(self.preference_combo)

        layout.addItem(vertical_spacer_2)  # Add spacer before narrative
        layout.addWidget(narrative_label)
        layout.addWidget(self.narrative_input)

        layout.addItem(vertical_spacer_3)  # Add spacer before submit button
        layout.addWidget(self.submit_button)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Connect the submit button to accept the dialog
        self.submit_button.clicked.connect(self.accept)

        self.setLayout(layout)

    def get_info(self):
        return self.name_input.text(), self.participant_combo.currentText(), self.preference_combo.currentText(), self.narrative_input.toPlainText()

    
class MessageReceiver(QObject):
    message_received = pyqtSignal(str)

    def __init__(self, socket):
        super().__init__()
        self.socket = socket

    def run(self):
        while True:
            try:
                data = self.socket.recv(1024).decode()
                self.message_received.emit(data)
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

class StorytellingClientGUI(QWidget):
    message_received = pyqtSignal(str)
    input_enabled = pyqtSignal(bool)
    connection_closed = pyqtSignal()

    def __init__(self, socket, story_words_with_weights, client_name):
        super().__init__()
        self.socket = socket
        self.selected_words = []
        self.used_words = set()
        self.points = 0
        self.story_words_with_weights = story_words_with_weights
        self.client_name = client_name
        self.initUI()
        self.setupThreads()

    def initUI(self):
        self.setWindowTitle(f'Storytelling Client: {self.client_name}')
        self.setFixedSize(1600, 900)

        main_layout = QVBoxLayout()

        # Top panel: Status, Points, CPU, Memory
        top_panel_layout = QHBoxLayout()

        self.status_label = QLabel("Welcome!", self)
        self.status_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(50)
        top_panel_layout.addWidget(self.status_label)

        self.points_label = QLabel(f'Points: {self.points}', self)
        self.points_label.setFont(QFont('Arial', 24, QFont.Weight.Bold))
        self.points_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.points_label.setFixedHeight(50)
        top_panel_layout.addWidget(self.points_label)


        main_layout.addLayout(top_panel_layout)

        # Split layout: left for storyboard and audio recorder, right for story display
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left half: Storyboard and audio recorder
        left_layout = QVBoxLayout()
        self.storyboard = StoryboardWidget(self.story_words_with_weights)
        self.storyboard.setEnabled(False)
        self.storyboard.word_selected_signal.connect(self.update_selected_words)
        left_layout.addWidget(self.storyboard, 2)

        self.audio_recorder = AudioRecorderWidget()
        self.audio_recorder.transcription_ready.connect(self.send_transcription)
        left_layout.addWidget(self.audio_recorder, 1)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Right half: Story display with fixed size
        self.story_display = QTextEdit()
        self.story_display.setReadOnly(True)
        self.story_display.setFont(QFont("Arial", 18))
        self.story_display.setFixedSize(1050, 700)
        self.story_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.story_display.setEnabled(True)

        self.selected_words_display = QLabel('Selected Words: ', self)
        self.selected_words_display.setFont(QFont("Arial", 16))
        self.selected_words_display.setFixedHeight(50)
        self.selected_words_display.setWordWrap(True)

        splitter.addWidget(left_widget)
        splitter.addWidget(self.story_display)
        splitter.setSizes([800, 800])

        main_layout.addWidget(splitter)
        main_layout.addWidget(self.selected_words_display)

        self.setLayout(main_layout)

    def setupThreads(self):
        # Message receiving thread
        self.receive_thread = QThread()
        self.receiver = MessageReceiver(self.socket)
        self.receiver.moveToThread(self.receive_thread)
        self.receive_thread.started.connect(self.receiver.run)
        self.receiver.message_received.connect(self.handle_message)
        self.receive_thread.start()

    def handle_message(self, data):
        print(f'Received data: {data}')
        if data.startswith("TELL"):
            self.input_enabled.emit(True)
        elif data.startswith("STORY:"):
            parts = data.split(":", 3)  # Split into at most 4 parts
            print(f'Printing parts: {parts}')
            if len(parts) == 3:  # STORY:name:content
                _, name, content = parts
                self.append_to_story_display(f"{name}: {content.strip()}")
            elif len(parts) == 4:  # STORY:Selected word:name:content
                _, name, selected_word, content = parts
                selected_word = selected_word.strip()  # Remove any whitespace
                if name.strip() != self.client_name.strip():
                    print(f'checking for client name')
                    if selected_word in [word.strip() for word in self.story_words_with_weights]:
                        points = int(self.story_words_with_weights[selected_word])
                        self.points += points
                        self.points_label.setText(f'Points: {self.points}')
                    self.cross_out_word(selected_word)
                self.append_to_story_display(f"{name}: "+ content.strip())
            else:
                print(f"Unexpected STORY format: {data}")
            self.socket.send("STORY_ACK".encode())
        elif data == "END":
            self.handle_end_message()
        else:
            pass
    
    def cross_out_word(self, word):
        word = word.strip()  # Remove any whitespace
        if word not in self.storyboard.crossed_words:
            self.storyboard.crossed_words.add(word)
            self.storyboard.update_button_styles()

    def handle_end_message(self):
        self.status_label.setText("Congratulations! The story has ended.")
        # self.input_enabled.emit(False)
        self.audio_recorder.record_button.setEnabled(False)
        self.storyboard.setEnabled(False)
        self.close_connection()
        self.connection_closed.emit()

    def close_connection(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        print("Connection closed")


    def append_to_story_display(self, text):
        self.story_display.append(text)
        cursor = self.story_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.story_display.setTextCursor(cursor)
        self.story_display.ensureCursorVisible()

    def update_selected_words(self, word):
        if word not in self.selected_words:
            self.selected_words.append(word)
            self.selected_words_display.setText(f'Selected Words: {", ".join(self.selected_words)}')
            # self.append_to_story_display(f"Selected word: {word}")
            # self.socket.send(f"WORD:{word}".encode())
            points = int(self.story_words_with_weights[word])
            self.points += points
            self.points_label.setText(f'Points: {self.points}')

    def send_transcription(self, transcription):
        message = ""
        if transcription.strip().lower() == ":empty:":
            transcription = "Please continue the story"

        
        unused_words = [word for word in self.selected_words if word not in self.used_words]
        
        if unused_words:
            # Use the first unused word
            word = unused_words[0]
            self.used_words.add(word)
            message = f"{word}:{transcription}"
        else:
            message = transcription

        self.socket.send(f"{message}".encode())
        self.input_enabled.emit(False)

    def on_input_enabled(self, enabled):
        self.audio_recorder.record_button.setEnabled(enabled)
        self.storyboard.setEnabled(enabled)
        self.status_label.setText("Your turn to continue the story" if enabled else "Waiting for your turn...")

    def update_resource_labels(self, cpu, memory):
        self.cpu_label.setText(f'CPU: {cpu:.1f}%')
        self.memory_label.setText(f'Memory: {memory:.1f}%')

    def closeEvent(self, event):
        if self.socket:
            self.socket.close()

        if self.receive_thread.isRunning():
                self.receive_thread.quit()
                self.receive_thread.wait()
        event.accept()

def connect_to_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, server_port))
    return sock

def read_story_words():

    try:
        with open('story_words.txt', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("story_words.txt file not found. Using default words.")

def main():
    app = QApplication(sys.argv)
    exception_handler = install_global_exception_handler(app)
    
    try:
        dialog = NameInputDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, participant_id, preference, narrative = dialog.get_info()

            print("Connecting to server...")
            client_socket = connect_to_server()
            print("Connected to server")

            user_info = f"{name}:{participant_id}:{preference}:{narrative}"
            print(f"Sending user info: {user_info}")
            client_socket.send(user_info.encode())

            word_weight_string = client_socket.recv(1024).decode()
            print(f'weight string: {word_weight_string}')
            story_words_with_weights = dict(item.split(':') for item in word_weight_string.split(','))
            print(f"Received story words with weights: {story_words_with_weights}")
            
            window = StorytellingClientGUI(client_socket, story_words_with_weights, name)
            window.status_label.setText(f"Welcome, {name}!")
            window.message_received.connect(window.handle_message)
            window.input_enabled.connect(window.on_input_enabled)
            window.show()

            sys.exit(app.exec())
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        if 'client_socket' in locals():
            client_socket.close()


if __name__ == '__main__':
    main()