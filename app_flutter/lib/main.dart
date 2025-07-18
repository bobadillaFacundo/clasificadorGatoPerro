import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const CatDogClassifierApp());
}

class CatDogClassifierApp extends StatelessWidget {
  const CatDogClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Clasificador Gato/Perro',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
        fontFamily: 'Arial',
      ),
      home: const ClassifierHomePage(),
    );
  }
}

class ClassifierHomePage extends StatefulWidget {
  const ClassifierHomePage({super.key});

  @override
  State<ClassifierHomePage> createState() => _ClassifierHomePageState();
}

class _ClassifierHomePageState extends State<ClassifierHomePage> {
  File? _image;
  String? _result;
  bool _loading = false;
  Interpreter? _interpreter;
  final picker = ImagePicker();
  double? _confidence;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      // Cargar el modelo directamente desde assets usando bytes
      final modelData = await rootBundle.load('assets/model.tflite');
      _interpreter = await Interpreter.fromBuffer(modelData.buffer.asUint8List());
    } catch (e) {
      setState(() {
        _result = 'Error cargando el modelo: $e';
        _loading = false;
      });
      print('Error cargando el modelo: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    if (source == ImageSource.camera && Platform.isWindows) {
      _showCameraNotSupported();
      return;
    }
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _result = null;
      });
      _classifyImage(File(pickedFile.path));
    }
  }

  void _showCameraNotSupported() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Cámara no soportada'),
        content: const Text('La cámara solo está disponible en dispositivos móviles.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Future<void> _classifyImage(File imageFile) async {
    setState(() {
      _loading = true;
    });
    try {
      final bytes = await imageFile.readAsBytes();
      img.Image? oriImage = img.decodeImage(bytes);
      if (oriImage == null) {
        setState(() {
          _result = 'No se pudo leer la imagen';
          _loading = false;
          _confidence = null;
        });
        return;
      }
      img.Image resized = img.copyResize(oriImage, width: 224, height: 224);
      var input = List.generate(224, (y) => List.generate(224, (x) => List.filled(3, 0.0)));
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          input[y][x][0] = pixel.r / 255.0;
          input[y][x][1] = pixel.g / 255.0;
          input[y][x][2] = pixel.b / 255.0;
        }
      }
      var inputBuffer = Float32List(1 * 224 * 224 * 3);
      int index = 0;
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          for (int c = 0; c < 3; c++) {
            inputBuffer[index++] = input[y][x][c];
          }
        }
      }
      var output = List.filled(2, 0.0).reshape([1, 2]);
      _interpreter?.run(inputBuffer.reshape([1, 224, 224, 3]), output);
      print('Output: ${output[0]}');
      int pred = output[0][0] > output[0][1] ? 0 : 1;
      double confidence = output[0][pred].toDouble();
      setState(() {
        _result = pred == 0 ? 'Gato 🐱' : 'Perro 🐶';
        _confidence = confidence;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _result = 'Error en la inferencia: $e';
        _loading = false;
        _confidence = null;
      });
      print('Error en la inferencia: $e');
    }
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isWindows = Platform.isWindows;
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFB2FEFA), Color(0xFF0ED2F7)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 20),
                  const Text(
                    'Clasificador de Gatos y Perros',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF22223B),
                      shadows: [
                        Shadow(
                          blurRadius: 8,
                          color: Colors.white,
                          offset: Offset(2, 2),
                        ),
                      ],
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 20),
                  Card(
                    elevation: 8,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Container(
                      width: 320,
                      height: 320,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(24),
                        color: Colors.white,
                      ),
                      child: _image != null
                          ? ClipRRect(
                              borderRadius: BorderRadius.circular(24),
                              child: Image.file(_image!, fit: BoxFit.cover, width: 320, height: 320),
                            )
                          : ClipRRect(
                              borderRadius: BorderRadius.circular(24),
                              child: Image.asset(
                                'image/fondo.png',
                                fit: BoxFit.cover,
                                width: 320,
                                height: 320,
                              ),
                            ),
                    ),
                  ),
                  const SizedBox(height: 30),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        icon: const Icon(Icons.photo, size: 32),
                        label: const Text('Galería', style: TextStyle(fontSize: 20)),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                      ),
                      if (!Platform.isWindows) ...[
                        const SizedBox(width: 20),
                        ElevatedButton.icon(
                          onPressed: () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt, size: 32),
                          label: const Text('Cámara', style: TextStyle(fontSize: 20)),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 30),
                  if (_loading)
                    const CircularProgressIndicator()
                  else if (_result != null)
                    Card(
                      elevation: 6,
                      color: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 20),
                        child: Column(
                          children: [
                            Text(
                              'Resultado: $_result',
                              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.teal),
                            ),
                            if (_confidence != null)
                              Padding(
                                padding: const EdgeInsets.only(top: 8.0),
                                child: Text(
                                  'Precisión: ${(_confidence! * 100).toStringAsFixed(2)}%',
                                  style: const TextStyle(fontSize: 20, color: Colors.black54),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(height: 40),
                  const Text(
                    'Powered by Flutter & TensorFlow Lite',
                    style: TextStyle(color: Colors.black38, fontSize: 16),
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
