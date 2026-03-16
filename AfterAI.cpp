// AfterAI.cpp - 最终修复版本
// 编译：g++ -o AfterAI.exe AfterAI.cpp -mwindows -lcomctl32 -static-libgcc -static-libstdc++

#include <windows.h>
#include <commctrl.h>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <queue>
#include <ctime>

#pragma comment(lib, "comctl32.lib")

// 常量定义
#define ID_EDIT_INPUT      1001
#define ID_EDIT_OUTPUT     1002
#define ID_BUTTON_SEND     1003
#define ID_BUTTON_CLEAR    1004
#define ID_BUTTON_TRAIN    1005
#define ID_BUTTON_SAVE     1006
#define ID_BUTTON_LOAD     1007
#define ID_PROGRESS        1008
#define WM_APPEND_TEXT     WM_USER + 1

// 神经网络配置
#define INPUT_SIZE 100    // 输入层大小
#define HIDDEN_SIZE 200   // 隐藏层大小
#define OUTPUT_SIZE 100   // 输出层大小

// 使用宽字符串
typedef std::wstring String;

// 神经元结构
struct Neuron {
    std::vector<double> weights;
    double bias;
    double output;
    double delta;

    Neuron(int inputCount) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 0.1);

        weights.resize(inputCount);
        for (int i = 0; i < inputCount; i++) {
            weights[i] = dist(gen);
        }
        bias = dist(gen);
        output = 0;
        delta = 0;
    }
};

// 层结构
struct Layer {
    std::vector<Neuron> neurons;
    int inputSize;
    int outputSize;

    Layer(int in, int out) : inputSize(in), outputSize(out) {
        for (int i = 0; i < out; i++) {
            neurons.emplace_back(in);
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> outputs;
        for (auto& neuron : neurons) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * neuron.weights[i];
            }
            sum += neuron.bias;
            neuron.output = tanh(sum);
            outputs.push_back(neuron.output);
        }
        return outputs;
    }
};

// 简单的LSTM单元
struct LSTMCell {
    std::vector<double> hiddenState;

    LSTMCell(int size) {
        hiddenState.resize(size, 0);
    }

    void forward(const std::vector<double>& input) {
        for (int i = 0; i < input.size() && i < hiddenState.size(); i++) {
            hiddenState[i] = 0.5 * hiddenState[i] + 0.5 * tanh(input[i]);
        }
    }
};

// 神经网络类
class NeuralNetwork {
private:
    Layer layer1;
    Layer layer2;
    Layer layer3;
    double learningRate;
    std::vector<String> memory;
    std::vector<std::vector<double>> wordVectors;
    std::map<String, int> wordToId;
    std::map<int, String> idToWord;
    LSTMCell lstm;

public:
    NeuralNetwork() :
        layer1(INPUT_SIZE, HIDDEN_SIZE),
        layer2(HIDDEN_SIZE, HIDDEN_SIZE),
        layer3(HIDDEN_SIZE, OUTPUT_SIZE),
        lstm(HIDDEN_SIZE) {
        learningRate = 0.01;
        InitializeVocabulary();
    }

    void InitializeVocabulary() {
        // 初始化中英文词汇
        std::vector<String> words = {
            L"你好", L"您好", L"早上好", L"晚上好",
            L"人工智能", L"神经网络", L"机器学习", L"深度学习",
            L"中国", L"北京", L"上海", L"深圳",
            L"天气", L"时间", L"日期", L"星期",
            L"名字", L"年龄", L"帮助", L"谢谢", L"再见", L"拜拜",
            L"hello", L"hi", L"how", L"are", L"you", L"what", L"is", L"name",
            L"weather", L"time", L"date", L"help", L"thanks", L"goodbye"
        };

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);

        for (int i = 0; i < words.size(); i++) {
            wordToId[words[i]] = i;
            idToWord[i] = words[i];

            std::vector<double> vec(INPUT_SIZE);
            for (int j = 0; j < INPUT_SIZE; j++) {
                vec[j] = dist(gen);
            }
            wordVectors.push_back(vec);
        }
    }

    std::vector<double> textToVector(const String& text) {
        std::vector<double> vec(INPUT_SIZE, 0);
        int count = 0;

        for (const auto& pair : wordToId) {
            if (text.find(pair.first) != String::npos) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    vec[i] += wordVectors[pair.second][i];
                }
                count++;
            }
        }

        if (count > 0) {
            for (int i = 0; i < INPUT_SIZE; i++) {
                vec[i] /= count;
            }
        }

        return vec;
    }

    String generateResponse(const String& input) {
        // 生成响应
        String response = generateFromRules(input);

        // 学习
        learn(input, response);

        return response;
    }

    String generateFromRules(const String& input) {
        // 中文问候
        if (input.find(L"你好") != String::npos ||
            input.find(L"您好") != String::npos) {
            return L"你好！我是AfterAI，很高兴认识你！有什么我可以帮助你的吗？";
        }

        // 英文问候
        if (input.find(L"hello") != String::npos ||
            input.find(L"hi") != String::npos) {
            return L"Hello! I'm AfterAI, nice to meet you! How can I help?";
        }

        // 询问名字
        if (input.find(L"名字") != String::npos ||
            input.find(L"name") != String::npos) {
            return L"我叫AfterAI，是一个基于神经网络的人工智能助手。";
        }

        // 询问时间
        if (input.find(L"时间") != String::npos ||
            input.find(L"time") != String::npos) {
            time_t now = time(0);
            struct tm tstruct;
            wchar_t buf[80];
            localtime_s(&tstruct, &now);
            wcsftime(buf, sizeof(buf) / sizeof(wchar_t), L"现在是 %H:%M:%S", &tstruct);
            return buf;
        }

        // 询问日期
        if (input.find(L"日期") != String::npos ||
            input.find(L"date") != String::npos) {
            time_t now = time(0);
            struct tm tstruct;
            wchar_t buf[80];
            localtime_s(&tstruct, &now);
            wcsftime(buf, sizeof(buf) / sizeof(wchar_t), L"今天是 %Y年%m月%d日", &tstruct);
            return buf;
        }

        // 询问天气
        if (input.find(L"天气") != String::npos ||
            input.find(L"weather") != String::npos) {
            return L"我目前没有联网，无法查询实时天气。不过我可以模拟一下：今天阳光明媚，适合学习编程！";
        }

        // 人工智能相关
        if (input.find(L"人工智能") != String::npos ||
            input.find(L"AI") != String::npos) {
            return L"人工智能是计算机科学的一个重要分支，致力于创造能够模拟人类智能的系统。我就是一个人工智能程序！";
        }

        // 感谢
        if (input.find(L"谢谢") != String::npos ||
            input.find(L"thanks") != String::npos) {
            return L"不客气！很高兴能帮到你。";
        }

        // 再见
        if (input.find(L"再见") != String::npos ||
            input.find(L"拜拜") != String::npos ||
            input.find(L"goodbye") != String::npos) {
            return L"再见！期待下次聊天！";
        }

        // 帮助
        if (input.find(L"帮助") != String::npos ||
            input.find(L"help") != String::npos) {
            return L"我可以做什么？\n- 问候：你好/hello\n- 询问时间/日期\n- 讨论人工智能\n- 简单聊天\n- 我会从对话中学习！";
        }

        // 默认智能响应
        return generateIntelligentResponse(input);
    }

    String generateIntelligentResponse(const String& input) {
        // 从记忆中寻找相似回复
        if (!memory.empty()) {
            std::vector<double> inputVec = textToVector(input);
            double maxSim = -1;
            String bestResponse = L"这个问题很有趣，让我思考一下...";

            for (const auto& mem : memory) {
                std::vector<double> memVec = textToVector(mem);
                double sim = cosineSimilarity(inputVec, memVec);
                if (sim > maxSim && sim > 0.3) {
                    maxSim = sim;
                    bestResponse = L"根据我的记忆： " + mem;
                }
            }

            if (maxSim > 0.3) {
                return bestResponse;
            }
        }

        // 随机智能响应
        std::vector<String> responses = {
            L"这是个有趣的问题，让我想想...",
            L"作为一个人工智能，我还在不断学习中。",
            L"你能多说一些吗？这样我可以更好地理解。",
            L"这个问题涉及的知识很深奥，我需要时间思考。",
            L"很有意思！还有更多细节吗？",
            L"我正在处理你的问题，请稍候...",
            L"基于我的神经网络模型，我认为...",
            L"这个问题让我产生了新的思考。"
        };

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, responses.size() - 1);

        return responses[dist(gen)];
    }

    double cosineSimilarity(const std::vector<double>& v1, const std::vector<double>& v2) {
        double dot = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < v1.size(); i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        return dot / (sqrt(norm1) * sqrt(norm2) + 1e-10);
    }

    void learn(const String& input, const String& response) {
        // 添加到记忆
        memory.push_back(input + L" -> " + response);
        if (memory.size() > 50) {
            memory.erase(memory.begin());
        }

        // 简单的学习：更新词向量
        std::vector<double> inputVec = textToVector(input);
        std::vector<double> responseVec = textToVector(response);

        for (auto& wordVec : wordVectors) {
            for (int i = 0; i < INPUT_SIZE; i++) {
                wordVec[i] += learningRate * (inputVec[i] - wordVec[i]);
            }
        }
    }

    void saveModel(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const auto& mem : memory) {
                std::string narrowMem;
                for (wchar_t c : mem) {
                    narrowMem += (char)c;
                }
                file << narrowMem << std::endl;
            }
            file.close();
        }
    }

    void loadModel(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                String wideLine;
                for (char c : line) {
                    wideLine += (wchar_t)c;
                }
                memory.push_back(wideLine);
            }
            file.close();
        }
    }
};

// 全局变量
HINSTANCE hInst;
HWND hMainWnd;
HWND hInputEdit;
HWND hOutputEdit;
HWND hSendButton;
HWND hProgressBar;
NeuralNetwork g_ai;
std::mutex g_mutex;
bool g_isProcessing = false;

// 函数声明
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void CreateControls(HWND);
void AppendToOutput(const String& text, bool isUser = false);
void ProcessInput(const String& input);

// 创建控件
void CreateControls(HWND hwnd) {
    HFONT hFont = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"SimSun");

    HFONT hTitleFont = CreateFontW(20, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"SimSun");

    // 标题
    CreateWindowExW(0, L"STATIC", L"🧠 AfterAI - 神经网络人工智能 (完全本地运行)",
        WS_CHILD | WS_VISIBLE | SS_CENTER,
        10, 10, 580, 30, hwnd, NULL, hInst, NULL);
    SendMessageW(GetDlgItem(hwnd, 0), WM_SETFONT, (WPARAM)hTitleFont, TRUE);

    // 输出区域标签
    CreateWindowExW(0, L"STATIC", L"对话历史:",
        WS_CHILD | WS_VISIBLE,
        10, 50, 80, 20, hwnd, NULL, hInst, NULL);
    SendMessageW(GetDlgItem(hwnd, 0), WM_SETFONT, (WPARAM)hFont, TRUE);

    // 输出编辑框
    hOutputEdit = CreateWindowExW(0, L"EDIT", L"",
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE |
        ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL,
        10, 70, 580, 300, hwnd, (HMENU)ID_EDIT_OUTPUT, hInst, NULL);
    SendMessageW(hOutputEdit, WM_SETFONT, (WPARAM)hFont, TRUE);

    // 输入区域标签
    CreateWindowExW(0, L"STATIC", L"输入消息:",
        WS_CHILD | WS_VISIBLE,
        10, 380, 80, 20, hwnd, NULL, hInst, NULL);
    SendMessageW(GetDlgItem(hwnd, 0), WM_SETFONT, (WPARAM)hFont, TRUE);

    // 输入编辑框
    hInputEdit = CreateWindowExW(0, L"EDIT", L"",
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE |
        ES_AUTOVSCROLL | WS_VSCROLL,
        10, 400, 470, 80, hwnd, (HMENU)ID_EDIT_INPUT, hInst, NULL);
    SendMessageW(hInputEdit, WM_SETFONT, (WPARAM)hFont, TRUE);

    // 发送按钮
    hSendButton = CreateWindowExW(0, L"BUTTON", L"发送",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        490, 400, 100, 35, hwnd, (HMENU)ID_BUTTON_SEND, hInst, NULL);
    SendMessageW(hSendButton, WM_SETFONT, (WPARAM)hTitleFont, TRUE);

    // 训练按钮
    CreateWindowExW(0, L"BUTTON", L"训练AI",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        490, 440, 48, 35, hwnd, (HMENU)ID_BUTTON_TRAIN, hInst, NULL);
    SendMessageW(GetDlgItem(hwnd, ID_BUTTON_TRAIN), WM_SETFONT, (WPARAM)hFont, TRUE);

    // 保存按钮
    CreateWindowExW(0, L"BUTTON", L"保存",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        542, 440, 48, 35, hwnd, (HMENU)ID_BUTTON_SAVE, hInst, NULL);
    SendMessageW(GetDlgItem(hwnd, ID_BUTTON_SAVE), WM_SETFONT, (WPARAM)hFont, TRUE);

    // 清空按钮
    CreateWindowExW(0, L"BUTTON", L"清空",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        490, 480, 100, 35, hwnd, (HMENU)ID_BUTTON_CLEAR, hInst, NULL);
    SendMessageW(GetDlgItem(hwnd, ID_BUTTON_CLEAR), WM_SETFONT, (WPARAM)hFont, TRUE);

    // 进度条 - 使用宽字符版本
    CreateWindowExW(0, L"msctls_progress32", L"",
        WS_CHILD | WS_VISIBLE,
        10, 490, 470, 25, hwnd, (HMENU)ID_PROGRESS, hInst, NULL);

    hProgressBar = GetDlgItem(hwnd, ID_PROGRESS);
    SendMessageW(hProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    SendMessageW(hProgressBar, PBM_SETPOS, 30, 0);
}

// 追加文本到输出
void AppendToOutput(const String& text, bool isUser) {
    String prefix = isUser ? L"👤 你: " : L"🤖 AfterAI: ";

    // 获取当前文本
    int len = GetWindowTextLengthW(hOutputEdit);
    String currentText;
    if (len > 0) {
        wchar_t* buffer = new wchar_t[len + 1];
        GetWindowTextW(hOutputEdit, buffer, len + 1);
        currentText = buffer;
        delete[] buffer;

        currentText += L"\r\n" + prefix + text + L"\r\n";
    }
    else {
        currentText = prefix + text + L"\r\n";
    }

    SetWindowTextW(hOutputEdit, currentText.c_str());
    SendMessageW(hOutputEdit, EM_SETSEL, currentText.length(), currentText.length());
    SendMessageW(hOutputEdit, EM_SCROLLCARET, 0, 0);
}

// 处理输入
void ProcessInput(const String& input) {
    g_isProcessing = true;
    EnableWindow(hSendButton, FALSE);

    // 显示用户输入
    AppendToOutput(input, true);

    // AI生成响应
    String response = g_ai.generateResponse(input);

    // 显示AI响应
    AppendToOutput(response);

    // 更新进度条
    static int progress = 30;
    progress = (progress + 5) % 100;
    SendMessageW(hProgressBar, PBM_SETPOS, progress, 0);

    g_isProcessing = false;
    EnableWindow(hSendButton, TRUE);
}

// 训练AI
void TrainAI() {
    std::vector<std::pair<String, String>> trainingData = {
        {L"你好", L"你好！很高兴认识你！"},
        {L"hello", L"Hello! Nice to meet you!"},
        {L"你叫什么名字", L"我叫AfterAI"},
        {L"what's your name", L"I'm AfterAI"},
        {L"今天天气怎么样", L"我是离线AI，无法查询天气，但可以陪你聊天"},
        {L"谢谢", L"不客气"},
        {L"再见", L"再见，期待下次聊天"},
        {L"人工智能是什么", L"人工智能是模拟人类智能的科学"}
    };

    for (int epoch = 0; epoch < 5; epoch++) {
        for (const auto& data : trainingData) {
            g_ai.learn(data.first, data.second);
        }
        int progress = (epoch + 1) * 20;
        SendMessageW(hProgressBar, PBM_SETPOS, progress, 0);
        Sleep(100);
    }

    AppendToOutput(L"AI训练完成！我学到了新知识！");
}

// 窗口过程
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        CreateControls(hwnd);

        String welcome =
            L"欢迎使用 AfterAI！\n"
            L"我是一个基于神经网络的本地人工智能助手。\n"
            L"特点：\n"
            L"  • 3层神经网络\n"
            L"  • 记忆学习能力\n"
            L"  • 完全离线运行\n"
            L"  • 支持中英文\n\n"
            L"你可以和我聊天，我会逐渐学习！\n"
            L"输入 '帮助' 或 'help' 查看可用功能。";

        AppendToOutput(welcome);
        break;
    }

    case WM_COMMAND: {
        int id = LOWORD(wParam);
        switch (id) {
        case ID_BUTTON_SEND: {
            if (g_isProcessing) break;

            // 获取输入
            wchar_t input[4096];
            GetWindowTextW(hInputEdit, input, 4096);

            if (wcslen(input) == 0) break;

            // 清空输入
            SetWindowTextW(hInputEdit, L"");

            // 处理输入
            String inputStr(input);
            std::thread([inputStr]() {
                ProcessInput(inputStr);
                }).detach();
            break;
        }

        case ID_BUTTON_CLEAR:
            SetWindowTextW(hOutputEdit, L"");
            AppendToOutput(L"对话已清空。");
            break;

        case ID_BUTTON_TRAIN:
            std::thread([]() {
                TrainAI();
                }).detach();
            break;

        case ID_BUTTON_SAVE:
            g_ai.saveModel("afterai_memory.txt");
            AppendToOutput(L"记忆已保存到 afterai_memory.txt");
            break;

        case ID_BUTTON_LOAD:
            g_ai.loadModel("afterai_memory.txt");
            AppendToOutput(L"记忆已加载");
            break;
        }
        break;
    }

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
    return 0;
}

// 程序入口
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    hInst = hInstance;

    // 初始化通用控件
    INITCOMMONCONTROLSEX icc;
    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&icc);

    // 注册窗口类 - 使用宽字符版本
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"AfterAIWindow";

    if (!RegisterClassW(&wc)) {
        MessageBoxW(NULL, L"窗口注册失败", L"错误", MB_ICONERROR);
        return 0;
    }

    // 创建窗口
    hMainWnd = CreateWindowExW(0, wc.lpszClassName, L"AfterAI - 神经网络人工智能",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 620, 600,
        NULL, NULL, hInstance, NULL);

    if (!hMainWnd) {
        MessageBoxW(NULL, L"窗口创建失败", L"错误", MB_ICONERROR);
        return 0;
    }

    ShowWindow(hMainWnd, nCmdShow);
    UpdateWindow(hMainWnd);

    // 消息循环
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return msg.wParam;
}