#include <iostream>
#include <cmath>         // for M_PI
#define _USE_MATH_DEFINES
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Dear ImGui
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// --- GLSL Shaders ---------------------------------------------------------
static const char* VERT_SHADER = R"(
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNorm;
layout(location=2) in vec2 inUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec3 vPos;
out vec3 vNorm;
out vec2 vUv;

void main(){
    vec4 world = model * vec4(inPos,1.0);
    vPos = world.xyz;
    vNorm = mat3(transpose(inverse(model))) * inNorm;
    vUv = inUV;
    gl_Position = proj * view * world;
}
)";

static const char* FRAG_SHADER = R"(
#version 330 core
in vec3 vPos;
in vec3 vNorm;
in vec2 vUv;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 camPos;
uniform vec3 lightColor;
uniform float roughness;
uniform float metallic;

// Just plain white/black
const vec3 albedoWhite = vec3(1.5);
const vec3 albedoBlack = vec3(0.05);

// 
// This function tries to place black patches at certain "center" UVs
// to mimic the black pentagons/hexagons of a soccer ball.
//
vec3 getBaseColor(vec2 uv){
    // You can move or add more centers for your pattern
    vec2 C[12] = vec2[](
        vec2(0.50, 0.50), // big center in the middle
        vec2(0.25, 0.70), vec2(0.75, 0.70),
        vec2(0.25, 0.30), vec2(0.75, 0.30),
        vec2(0.10, 0.50), vec2(0.90, 0.50),
        vec2(0.50, 0.10), vec2(0.50, 0.90),
        // Additional random centers
        vec2(0.15, 0.15), vec2(0.85, 0.85), vec2(0.15, 0.85)
    );
    for(int i = 0; i < 12; i++){
        // radius 0.15 to get a bit bigger black patches
        if(distance(uv, C[i]) < 0.15) {
            return albedoBlack;
        }
    }
    return albedoWhite;
}

// -- PBR Utilities ----------------------------------------------------------

// Implementation of standard microfacet PBR (Cook-Torrance w/ GGX).
float DistributionGGX(vec3 N, vec3 H, float a){
    float a2 = a*a;
    float NdotH = max(dot(N,H),0.0);
    float denom = (NdotH*NdotH*(a2 - 1.0) + 1.0);
    return a2 / (3.14159 * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float k){
    return NdotV / (NdotV*(1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float k){
    float NdotV = max(dot(N,V), 0.0);
    float NdotL = max(dot(N,L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, k);
    float ggx1  = GeometrySchlickGGX(NdotL, k);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0){
    return F0 + (1.0 - F0)*pow(1.0 - cosTheta, 5.0);
}

void main(){
    vec3 N = normalize(vNorm);
    vec3 V = normalize(camPos - vPos);
    vec3 L = normalize(lightPos - vPos);
    vec3 H = normalize(V + L);

    // For more realistic geometry, you'd build an actual soccer mesh
    // with pentagon & hexagon faces. Here, we do "patches in UV" as a simpler approach.
    vec3 albedo = getBaseColor(vUv);

    // The base reflectivity at normal incidence
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance terms
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 nom   = D * G * F;
    float denom = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3 specular = nom / denom;

    // kD is the diffuse component
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    // Lambertian
    vec3 diffuse = kD * albedo / 3.14159;

    float NdotL = max(dot(N,L), 0.0);
    vec3 color = (diffuse + specular) * lightColor * NdotL;

    FragColor = vec4(color, 1.0);
}
)";

// ----------------------------------------------------------------------------
// Build a UV sphere mesh
// ----------------------------------------------------------------------------
void buildSphere(int sectors, int stacks,
                 std::vector<float> &vbo,
                 std::vector<unsigned> &ebo)
{
    // We assume M_PI is defined or we do #define M_PI 3.14159265358979323846
    // UV range: u = [0..1], v = [0..1]
    for(int i = 0; i <= stacks; i++){
        // phi from +π/2 down to -π/2
        float phi = M_PI/2.0f - i*(M_PI / (float)stacks);
        float y = sin(phi);
        float r = cos(phi);
        for(int j = 0; j <= sectors; j++){
            float theta = j*(2.0f*M_PI / (float)sectors);
            float x = r * cos(theta);
            float z = r * sin(theta);

            // position
            vbo.push_back(x);
            vbo.push_back(y);
            vbo.push_back(z);

            // normal
            vbo.push_back(x);
            vbo.push_back(y);
            vbo.push_back(z);

            // uv
            float u = j / (float)sectors;   // [0..1]
            float v = i / (float)stacks;    // [0..1]
            vbo.push_back(u);
            vbo.push_back(v);
        }
    }
    for(int i = 0; i < stacks; i++){
        for(int j = 0; j < sectors; j++){
            int a = i*(sectors+1) + j;
            int b = a + sectors + 1;
            ebo.push_back(a);
            ebo.push_back(b);
            ebo.push_back(a+1);

            ebo.push_back(b);
            ebo.push_back(b+1);
            ebo.push_back(a+1);
        }
    }
}

// ----------------------------------------------------------------------------
// Helper to compile a shader
// ----------------------------------------------------------------------------
GLuint compileShader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char buf[512];
        glGetShaderInfoLog(s, 512, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << "\n";
    }
    return s;
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------
int main(){
    // Initialize GLFW
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW.\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    GLFWwindow* win = glfwCreateWindow(800,600,"Soccer Ball PBR", nullptr, nullptr);
    if(!win){
        std::cerr << "Failed to create GLFW window.\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);

    // GLAD
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    // set viewport + handle window resize
    int width, height;
    glfwGetFramebufferSize(win, &width, &height);
    glViewport(0,0,width,height);
    glfwSetFramebufferSizeCallback(win, [](GLFWwindow*, int w, int h){
        glViewport(0,0,w,h);
    });

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Compile/link shaders
    GLuint vs = compileShader(GL_VERTEX_SHADER,   VERT_SHADER);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, FRAG_SHADER);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    // Build sphere geometry
    std::vector<float>    sphereVBO;
    std::vector<unsigned> sphereEBO;
    buildSphere(64, 64, sphereVBO, sphereEBO);

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind and upload
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVBO.size()*sizeof(float), sphereVBO.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereEBO.size()*sizeof(unsigned), sphereEBO.data(), GL_STATIC_DRAW);

    // positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    // UV
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    // Grab uniform locations
    GLint u_model  = glGetUniformLocation(prog, "model");
    GLint u_view   = glGetUniformLocation(prog, "view");
    GLint u_proj   = glGetUniformLocation(prog, "proj");
    GLint u_lightP = glGetUniformLocation(prog, "lightPos");
    GLint u_camP   = glGetUniformLocation(prog, "camPos");
    GLint u_lightC = glGetUniformLocation(prog, "lightColor");
    GLint u_rough  = glGetUniformLocation(prog, "roughness");
    GLint u_metal  = glGetUniformLocation(prog, "metallic");

    // Setup a projection matrix
    glUseProgram(prog);
    glm::mat4 projM = glm::perspective(glm::radians(45.0f),
                                       (float)width / (float)height,
                                       0.1f, 100.0f);
    glUniformMatrix4fv(u_proj, 1, GL_FALSE, glm::value_ptr(projM));

    // Light
    glUniform3f(u_lightP, 5.0f, 5.0f, 5.0f);
    glUniform3f(u_lightC, 1.0f, 1.0f, 1.0f);

    // Tweakable
    float roughness = 0.4f;
    float metallic  = 0.0f;

    // State settings
    glEnable(GL_DEPTH_TEST);
    // If sphere’s normals are inverted, disabling cull or reversing it can help
    glDisable(GL_CULL_FACE);

    // Main loop
    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // UI
        ImGui::Begin("Material Controls");
        ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f);
        ImGui::SliderFloat("Metallic", &metallic,   0.0f, 1.0f);
        ImGui::End();

        // Rendering
        glClearColor(0.1f,0.12f,0.15f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);

        // Camera
        glm::vec3 camPos(0.0f, 0.0f, 5.0f);
        glm::mat4 viewM = glm::lookAt(camPos, glm::vec3(0,0,0), glm::vec3(0,1,0));
        glUniformMatrix4fv(u_view, 1, GL_FALSE, glm::value_ptr(viewM));
        glUniform3f(u_camP, camPos.x, camPos.y, camPos.z);

        // Model (rotate over time)
        static float angle = 0.0f;
        angle += 0.005f;
        glm::mat4 modelM = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0,1,0));
        glUniformMatrix4fv(u_model, 1, GL_FALSE, glm::value_ptr(modelM));

        // Update roughness/metallic
        glUniform1f(u_rough, roughness);
        glUniform1f(u_metal, metallic);

        // Draw
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)sphereEBO.size(), GL_UNSIGNED_INT, 0);

        // Render ImGui on top
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
