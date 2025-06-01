// ---------------------------------------------------------------------------
// main.cpp — Render ball, wall, goal from CSV wavepoints, matching physical scale
//            and applying foot offset + orientation so they appear more "correct".
//            + Auto-play frames from wavepoints_*.csv
// ---------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
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

// stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// TinyOBJLoader
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Shaders
// ---------------------------------------------------------------------------
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

void main()
{
    vec4 worldPos = model * vec4(inPos,1.0);
    vPos  = worldPos.xyz;
    vNorm = mat3(transpose(inverse(model))) * inNorm;
    vUv   = inUV;
    gl_Position = proj * view * worldPos;
}
)";

static const char* FRAG_SHADER = R"(
#version 330 core
in  vec3 vPos;
in  vec3 vNorm;
in  vec2 vUv;
out vec4 FragColor;

uniform vec3  lightPos;
uniform vec3  camPos;
uniform vec3  lightColor;
uniform float roughness;
uniform float metallic;
uniform float exposure;

uniform int objectType; 
// 0=grass, 1=ball, 2=goal, 3=wall

// Grass
uniform sampler2D grass_albedo;
uniform sampler2D grass_normal;
uniform sampler2D grass_roughness;

// Ball
uniform sampler2D ball_albedo;
uniform sampler2D ball_normal;
uniform sampler2D ball_roughness;

// Goal
uniform sampler2D goal_albedo;
uniform sampler2D goal_normal;
uniform sampler2D goal_roughness;

// Wall
uniform sampler2D wall_albedo;
uniform sampler2D wall_normal;
uniform sampler2D wall_roughness;

// A simple PBR using GGX
float D_GGX(vec3 N, vec3 H, float a)
{
    float a2 = a*a;
    float NdotH= max(dot(N,H), 0.0);
    float denom= (NdotH*NdotH*(a2-1.0)+1.0);
    return a2/(3.14159* denom* denom);
}
float G_Schlick(float NdotV,float k)
{
    return NdotV/(NdotV*(1.0-k)+k);
}
float G_Smith(vec3 N,vec3 V,vec3 L,float a)
{
    float k=(a+1.0);
    k=(k*k)/8.0;
    float NdotV= max(dot(N,V),0.0);
    float NdotL= max(dot(N,L),0.0);
    return G_Schlick(NdotV,k)* G_Schlick(NdotL,k);
}
vec3 F_Schlick(float cosT, vec3 F0)
{
    return F0 + (1.0-F0)* pow((1.0-cosT),5.0);
}

void main()
{
    vec3 N= normalize(vNorm);
    vec3 V= normalize(camPos- vPos);
    vec3 L= normalize(lightPos- vPos);
    vec3 H= normalize(V+L);

    vec3 baseColor= vec3(1.0);
    float rVal=0.3;
    float mVal=0.0;
    vec3 nMap= vec3(0,0,1);

    if(objectType==0)
    {
        // grass
        baseColor= texture(grass_albedo,vUv*50.0).rgb;
        nMap     = texture(grass_normal,vUv*50.0).rgb*2.0 -1.0;
        rVal     = texture(grass_roughness,vUv*50.0).r;
    }
    else if(objectType==1)
    {
        // ball
        baseColor= texture(ball_albedo,vUv).rgb;
        nMap     = texture(ball_normal,vUv).rgb*2.0 -1.0;
        rVal     = texture(ball_roughness,vUv).r;
    }
    else if(objectType==2)
    {
        // goal
        baseColor= texture(goal_albedo,vUv).rgb;
        nMap     = texture(goal_normal,vUv).rgb*2.0 -1.0;
        rVal     = texture(goal_roughness,vUv).r;
    }
    else
    {
        // wall
        baseColor= texture(wall_albedo,vUv).rgb;
        nMap     = texture(wall_normal,vUv).rgb*2.0 -1.0;
        rVal     = texture(wall_roughness,vUv).r;
    }

    // mix user roughness, metallic
    rVal = mix(rVal, roughness, 0.5);
    mVal = mix(mVal, metallic, 0.2);

    // combine normal
    N= normalize(N+ nMap*0.5);

    // GGX
    vec3 F0= mix(vec3(0.04), baseColor, mVal);
    vec3 F = F_Schlick( max(dot(H,V),0.0), F0);
    float D= D_GGX(N,H,rVal);
    float G= G_Smith(N,V,L,rVal);

    vec3 specNumer= D*G*F;
    float denom= 4.0* max(dot(N,V),0.0)* max(dot(N,L),0.0)+0.0001;
    vec3 spec= specNumer/ denom;

    vec3 kd= (1.0-F)*(1.0- mVal);
    vec3 diff= kd* baseColor/ 3.14159;

    vec3 outColor= (diff+ spec)* max(dot(N,L),0.0)* lightColor;
    outColor= exposure*(0.2*baseColor+ outColor);

    FragColor= vec4(outColor,1.0);
}
)";

// ---------------------------------------------------------------------------
// 2. Utility Helpers
// ---------------------------------------------------------------------------
struct MeshGL
{
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    GLsizei indexCount;
};

struct BoundingBox
{
    glm::vec3 vmin;
    glm::vec3 vmax;
};

struct CSVLine
{
    float x,y,z;    
    std::string objectType;
    float width, height;
    float oriX, oriY, oriZ; 
};

bool loadObjMesh(const std::string& filename,
    const std::string& mtlDir,
    MeshGL& outMesh)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret= tinyobj::LoadObj(&attrib,&shapes,&materials,&warn,&err,
                    filename.c_str(),
                    mtlDir.empty()? nullptr:mtlDir.c_str());
    if(!warn.empty()) std::cout<<"TinyOBJ warn: "<<warn<<"\n";
    if(!err.empty())  std::cerr<<"TinyOBJ err:  "<<err<<"\n";
    if(!ret) return false;
    if(shapes.empty()){
        std::cerr<<"No shapes in "<<filename<<"\n";
        return false;
    }

    std::vector<float> vertexData;
    std::vector<unsigned int> indexData;
    size_t idxBase=0;

    for(const auto& shape : shapes)
    {
        for(size_t f=0; f< shape.mesh.indices.size(); ++f)
        {
            tinyobj::index_t idx= shape.mesh.indices[f];
            float px= attrib.vertices[3*idx.vertex_index+0];
            float py= attrib.vertices[3*idx.vertex_index+1];
            float pz= attrib.vertices[3*idx.vertex_index+2];

            float nx=0.f, ny=1.f, nz=0.f;
            if(idx.normal_index>=0)
            {
                nx= attrib.normals[3* idx.normal_index+0];
                ny= attrib.normals[3* idx.normal_index+1];
                nz= attrib.normals[3* idx.normal_index+2];
            }
            float u=0.f, v=0.f;
            if(idx.texcoord_index>=0)
            {
                u= attrib.texcoords[2* idx.texcoord_index+0];
                v= attrib.texcoords[2* idx.texcoord_index+1];
            }

            vertexData.push_back(px);
            vertexData.push_back(py);
            vertexData.push_back(pz);
            vertexData.push_back(nx);
            vertexData.push_back(ny);
            vertexData.push_back(nz);
            vertexData.push_back(u);
            vertexData.push_back(v);

            indexData.push_back((unsigned int)(idxBase+ f));
        }
        idxBase+= shape.mesh.indices.size();
    }

    glGenVertexArrays(1,&outMesh.vao);
    glGenBuffers(1,&outMesh.vbo);
    glGenBuffers(1,&outMesh.ebo);

    glBindVertexArray(outMesh.vao);

    glBindBuffer(GL_ARRAY_BUFFER, outMesh.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 vertexData.size()* sizeof(float),
                 vertexData.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, outMesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indexData.size()* sizeof(unsigned int),
                 indexData.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8* sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8* sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8* sizeof(float),(void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    outMesh.indexCount= (GLsizei) indexData.size();
    glBindVertexArray(0);

    return true;
}

BoundingBox measureMesh(const MeshGL& mesh)
{
    BoundingBox box;
    box.vmin= glm::vec3(1e9f);
    box.vmax= glm::vec3(-1e9f);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    GLint bufSize=0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER,GL_BUFFER_SIZE,&bufSize);
    if(bufSize<=0) return box;

    const float* vb= (const float*) glMapBuffer(GL_ARRAY_BUFFER,GL_READ_ONLY);
    if(!vb) return box;

    size_t stride= 8;
    size_t vertCount= size_t(bufSize)/ (stride* sizeof(float));
    for(size_t i=0; i<vertCount;++i)
    {
        float px= vb[i*stride+0];
        float py= vb[i*stride+1];
        float pz= vb[i*stride+2];
        box.vmin= glm::min(box.vmin, glm::vec3(px,py,pz));
        box.vmax= glm::max(box.vmax, glm::vec3(px,py,pz));
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);

    return box;
}

GLuint loadTex(const char* path)
{
    stbi_set_flip_vertically_on_load(true);
    int w,h,c;
    unsigned char* d= stbi_load(path,&w,&h,&c,0);
    if(!d){
        std::cerr<<"Texture fail: "<<path<<"\n";
        return 0;
    }
    GLenum fmt= (c==1?GL_RED: (c==3?GL_RGB: GL_RGBA));

    GLuint tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexImage2D(GL_TEXTURE_2D,0,fmt,w,h,0,fmt,GL_UNSIGNED_BYTE,d);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

    stbi_image_free(d);
    return tex;
}

GLuint compileShader(GLenum type,const char* src)
{
    GLuint s= glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok)
    {
        char log[512];
        glGetShaderInfoLog(s,512,nullptr,log);
        std::cerr<<"Shader error:\n"<<log<<"\n";
    }
    return s;
}

// reading CSV
std::vector<CSVLine> g_ballFrames;
CSVLine g_wallData;
CSVLine g_goalData;

// The current frame we’re displaying
int g_currentFrame = 0;

// camera & PBR
bool  g_followCam= false;
float g_rough   = 0.3f;
float g_metal   = 0.f;
float g_exposure= 1.5f;

// We'll store plane, ball, wall, goal
MeshGL g_planeMesh;
MeshGL g_ballMesh;
MeshGL g_wallMesh;
MeshGL g_goalMesh;

// bounding boxes
BoundingBox g_ballBox;
BoundingBox g_wallBox;
BoundingBox g_goalBox;

// textures
GLuint grassDiff, grassNorm, grassRgh;
GLuint ballDiff,  ballNorm,  ballRgh;
GLuint goalDiff,  goalNorm,  goalRgh;
GLuint wallDiff,  wallNorm,  wallRgh;

bool loadCSV(const std::string& csvPath)
{
    std::ifstream ifs(csvPath);
    if(!ifs.is_open()){
        std::cerr << "Cannot open " << csvPath << "\n";
        return false;
    }
    std::string line;
    bool skipHeader=true;
    while(std::getline(ifs,line))
    {
        if(skipHeader){ 
            skipHeader=false; 
            continue; 
        } // skip first line

        std::stringstream ss(line);
        std::vector<std::string> token;
        std::string tk;
        while(std::getline(ss,tk,',')) token.push_back(tk);
        if(token.size()<9) continue;

        CSVLine row;
        row.x= std::stof(token[0]);
        row.y= std::stof(token[1]);
        row.z= std::stof(token[2]);
        row.objectType= token[3];
        row.width = std::stof(token[4]);
        row.height= std::stof(token[5]);
        row.oriX  = std::stof(token[6]);
        row.oriY  = std::stof(token[7]);
        row.oriZ  = std::stof(token[8]);

        if(row.objectType=="ball"){
            g_ballFrames.push_back(row);
        }
        else if(row.objectType=="wall"){
            g_wallData= row;
        }
        else if(row.objectType=="goal"){
            g_goalData= row;
        }
    }
    std::cout << "Loaded CSV: ball=" << g_ballFrames.size()
              << ", wall=(" << g_wallData.x <<","<< g_wallData.y <<","<< g_wallData.z <<")"
              << " w="<< g_wallData.width <<" h="<< g_wallData.height
              << ", goal=(" << g_goalData.x <<","<< g_goalData.y <<","<< g_goalData.z <<")"
              << " w="<< g_goalData.width <<" h="<< g_goalData.height
              << "\n";
    return true;
}

// build plane
void buildPlane(MeshGL& mesh,float s=100.f)
{
    float vb[]={
        -s,0,-s,   0,1,0,  0,0,
         s,0,-s,   0,1,0,  1,0,
         s,0, s,   0,1,0,  1,1,
        -s,0, s,   0,1,0,  0,1
    };
    unsigned int ib[]={0,1,2, 2,3,0};

    glGenVertexArrays(1,&mesh.vao);
    glGenBuffers(1,&mesh.vbo);
    glGenBuffers(1,&mesh.ebo);

    glBindVertexArray(mesh.vao);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vb), vb, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ib), ib, GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    mesh.indexCount=6;
}

int main()
{
    // GLFW init
    if(!glfwInit()){
        std::cerr<<"glfw init fail\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    GLFWwindow* win= glfwCreateWindow(1280,720,"CSV Demo (Auto-play frames)",nullptr,nullptr);
    if(!win){
        std::cerr<<"Failed create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr<<"Fail init GLAD\n";
        return -1;
    }
    glfwSwapInterval(1);

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win,true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // compile shaders
    GLuint vs= compileShader(GL_VERTEX_SHADER,   VERT_SHADER);
    GLuint fs= compileShader(GL_FRAGMENT_SHADER, FRAG_SHADER);
    GLuint prog= glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    // plane
    buildPlane(g_planeMesh);

    // load ball, measure
    if(!loadObjMesh("assets/soccer_ball.obj","assets/", g_ballMesh)){
        std::cerr<<"Fail soccer_ball.obj\n";
        return -1;
    }
    g_ballBox= measureMesh(g_ballMesh);

    // load goal
    if(!loadObjMesh("assets/goal.obj","assets/", g_goalMesh)){
        std::cerr<<"Fail goal.obj\n";
        return -1;
    }
    g_goalBox= measureMesh(g_goalMesh);

    // load wall
    if(!loadObjMesh("assets/Man.obj","assets/", g_wallMesh)){
        std::cerr<<"Fail Man.obj\n";
        return -1;
    }
    g_wallBox= measureMesh(g_wallMesh);

    // load textures
    grassDiff= loadTex("assets/grass_diffuse.png");
    grassNorm= loadTex("assets/grass_normal.png");
    grassRgh = loadTex("assets/grass_roughness.png");

    ballDiff = loadTex("assets/ball_diffuse.jpg");
    ballNorm = loadTex("assets/ball_normal.jpg");
    ballRgh  = loadTex("assets/ball_roughness.jpg");

    goalDiff = loadTex("assets/goal_diffuse.png");
    goalNorm = loadTex("assets/goal_normal.png");
    goalRgh  = loadTex("assets/goal_roughness.png");

    wallDiff = loadTex("assets/Man_BaseColor.png");
    wallNorm = loadTex("assets/Man_Normal.png");
    wallRgh  = loadTex("assets/Man_Roughness.png");

    // set up uniforms
    glUseProgram(prog);
    GLint uM   = glGetUniformLocation(prog,"model");
    GLint uV   = glGetUniformLocation(prog,"view");
    GLint uP   = glGetUniformLocation(prog,"proj");
    GLint uLP  = glGetUniformLocation(prog,"lightPos");
    GLint uLC  = glGetUniformLocation(prog,"lightColor");
    GLint uCP  = glGetUniformLocation(prog,"camPos");
    GLint uR   = glGetUniformLocation(prog,"roughness");
    GLint uMet = glGetUniformLocation(prog,"metallic");
    GLint uExp = glGetUniformLocation(prog,"exposure");
    GLint uObj = glGetUniformLocation(prog,"objectType");

    // light
    glUniform3f(uLP,0.0f,30.0f,30.0f);
    glUniform3f(uLC,3.0f,3.0f,3.0f);

    // sampler
    glUniform1i(glGetUniformLocation(prog,"grass_albedo"),    0);
    glUniform1i(glGetUniformLocation(prog,"grass_normal"),    1);
    glUniform1i(glGetUniformLocation(prog,"grass_roughness"), 2);

    glUniform1i(glGetUniformLocation(prog,"ball_albedo"),     3);
    glUniform1i(glGetUniformLocation(prog,"ball_normal"),     4);
    glUniform1i(glGetUniformLocation(prog,"ball_roughness"),  5);

    glUniform1i(glGetUniformLocation(prog,"goal_albedo"),     6);
    glUniform1i(glGetUniformLocation(prog,"goal_normal"),     7);
    glUniform1i(glGetUniformLocation(prog,"goal_roughness"),  8);

    glUniform1i(glGetUniformLocation(prog,"wall_albedo"),     9);
    glUniform1i(glGetUniformLocation(prog,"wall_normal"),     10);
    glUniform1i(glGetUniformLocation(prog,"wall_roughness"),  11);

    // projection
    int wScr,hScr;
    glfwGetFramebufferSize(win,&wScr,&hScr);
    glm::mat4 projMat= glm::perspective(glm::radians(45.f), float(wScr)/float(hScr), 0.1f, 500.f);
    glUniformMatrix4fv(uP,1,GL_FALSE, glm::value_ptr(projMat));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // load wavepoints CSV
    if(!loadCSV("output/wavepoints_1.csv")){
        std::cerr << "Check wavepoints_1.csv\n";
    }
    // If no ball frames found, add a default
    if(g_ballFrames.empty()){
        g_ballFrames.push_back({0,0,0,"ball",1,1,0,0,0});
    }

    // g_goalData.z    = -30.0f;
    // g_goalData.oriY = 180.0f;

    float frameInterval = 0.016f; 
    float timeAccum     = 0.0f;

    // main loop
    double lastTime = glfwGetTime();
    while(!glfwWindowShouldClose(win))
    {
        double now = glfwGetTime();
        float dt   = float(now - lastTime);
        lastTime   = now;

        glfwPollEvents();

        // ---------------------------
        //  AUTO-PLAY FRAMES
        // ---------------------------
        timeAccum += dt;
        if(timeAccum >= frameInterval)
        {
            timeAccum = 0.0f;
            g_currentFrame++;
            // 不循环：到最后一帧就停住
            if(g_currentFrame >= (int)g_ballFrames.size())
                g_currentFrame = (int)g_ballFrames.size()-1;
        }

        // ---------------------------
        //  Setup ImGui
        // ---------------------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::Text("Frame index: %d / %d", g_currentFrame, (int)g_ballFrames.size()-1);
        if(ImGui::Button("Reset to 0")) {
            g_currentFrame = 0;
            timeAccum      = 0.0f;
        }
        ImGui::Checkbox("Follow cam", &g_followCam);
        ImGui::SliderFloat("Exposure", &g_exposure, 0.1f, 5.f);
        ImGui::SliderFloat("Roughness",&g_rough,    0.f,   1.f);
        ImGui::SliderFloat("Metallic", &g_metal,    0.f,   1.f);

        // We can show some debug info about the ball
        if(!g_ballFrames.empty()) {
            CSVLine& ballLine = g_ballFrames[g_currentFrame];
            ImGui::SeparatorText("Ball Debug");
            ImGui::Text("Ball CSV: (%.2f, %.2f, %.2f)", 
                        ballLine.x, ballLine.y, ballLine.z);
        }

        ImGui::End();

        // get the current ball data
        CSVLine& ballLine = g_ballFrames[g_currentFrame];

        // camera
        glm::vec3 camPos, lookTarget(0,0,0);
        if(g_followCam)
        {
            // follow the ball
            camPos = glm::vec3(ballLine.x, ballLine.y, ballLine.z) + glm::vec3(0,2,10);
            lookTarget = glm::vec3(ballLine.x, ballLine.y, ballLine.z);
        }
        else
        {
            // fixed vantage
            camPos = glm::vec3(5.0f, 8.0f, -40.0f);
            lookTarget = glm::vec3(0, 0, 0);
        }

        glm::mat4 view = glm::lookAt(camPos, lookTarget, glm::vec3(0,1,0));

        glfwGetFramebufferSize(win,&wScr,&hScr);
        glViewport(0,0,wScr,hScr);
        glClearColor(0.25f,0.3f,0.35f,1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // use program
        glUseProgram(prog);
        glUniformMatrix4fv(uV,1,GL_FALSE, glm::value_ptr(view));
        glUniform3f(uCP, camPos.x, camPos.y, camPos.z);
        glUniform1f(uR,  g_rough);
        glUniform1f(uMet,g_metal);
        glUniform1f(uExp,g_exposure);

        // bind textures
        glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, grassDiff);
        glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, grassNorm);
        glActiveTexture(GL_TEXTURE2);  glBindTexture(GL_TEXTURE_2D, grassRgh);

        glActiveTexture(GL_TEXTURE3);  glBindTexture(GL_TEXTURE_2D, ballDiff);
        glActiveTexture(GL_TEXTURE4);  glBindTexture(GL_TEXTURE_2D, ballNorm);
        glActiveTexture(GL_TEXTURE5);  glBindTexture(GL_TEXTURE_2D, ballRgh);

        glActiveTexture(GL_TEXTURE6);  glBindTexture(GL_TEXTURE_2D, goalDiff);
        glActiveTexture(GL_TEXTURE7);  glBindTexture(GL_TEXTURE_2D, goalNorm);
        glActiveTexture(GL_TEXTURE8);  glBindTexture(GL_TEXTURE_2D, goalRgh);

        glActiveTexture(GL_TEXTURE9);  glBindTexture(GL_TEXTURE_2D, wallDiff);
        glActiveTexture(GL_TEXTURE10); glBindTexture(GL_TEXTURE_2D, wallNorm);
        glActiveTexture(GL_TEXTURE11); glBindTexture(GL_TEXTURE_2D, wallRgh);

        GLint uM   = glGetUniformLocation(prog,"model");
        GLint uObj = glGetUniformLocation(prog,"objectType");

        // --------------------------
        // draw plane (grass)
        // --------------------------
        glUniform1i(uObj, 0); // grass
        {
            glm::mat4 m(1.f);
            glUniformMatrix4fv(uM,1,GL_FALSE, glm::value_ptr(m));
            glBindVertexArray(g_planeMesh.vao);
            glDrawElements(GL_TRIANGLES,g_planeMesh.indexCount,GL_UNSIGNED_INT,0);
        }

        // --------------------------
        // draw wall (4-person)
        // --------------------------
        glUniform1i(uObj,3); // wall
        {
            glm::vec3 boxSize= g_wallBox.vmax - g_wallBox.vmin;
            float objWidth  = boxSize.x;
            float objHeight = boxSize.y;

            float scaleX = (objWidth  > 1e-5f) ? (g_wallData.width  / objWidth)  : 1.f;
            float scaleY = (objHeight > 1e-5f) ? (g_wallData.height / objHeight) : 1.f;
            float scaleZ = scaleX;

            float yawRad = glm::radians(g_wallData.oriY + 180.0f);
            glm::vec3 dir = glm::vec3(cos(yawRad), 0.f, -sin(yawRad));  
            glm::vec3 center = glm::vec3(g_wallData.x, g_wallData.y, g_wallData.z);

            // total # of persons
            const int wallCount = 4;
            float spacing = g_wallData.width / (float)wallCount;

            for (int i = 0; i < wallCount; ++i)
            {
                float local = (i + 0.5f)* spacing - g_wallData.width * 0.5f;
                glm::vec3 pos = center + dir * local;

                glm::mat4 mWall(1.f);
                mWall = glm::translate(mWall, pos);
                mWall = glm::rotate(mWall, yawRad, glm::vec3(0,1,0));
                mWall = glm::scale(mWall, glm::vec3(scaleX, scaleY, scaleZ));

                glUniformMatrix4fv(uM,1,GL_FALSE, glm::value_ptr(mWall));
                glBindVertexArray(g_wallMesh.vao);
                glDrawElements(GL_TRIANGLES, g_wallMesh.indexCount, GL_UNSIGNED_INT, nullptr);
            }
        }

        // --------------------------
        // draw ball
        // --------------------------
        {
            glUniform1i(uObj, 1); // ball
            glm::vec3 bSize = g_ballBox.vmax - g_ballBox.vmin;
            float sx= (bSize.x > 1e-5f) ? (ballLine.width  / bSize.x) * 0.8f : 1.f;
            float sy= (bSize.y > 1e-5f) ? (ballLine.height / bSize.y) * 0.8f: 1.f;
            float sz= sx; // keep sphere scale

            float yOffset = 0.24f; // offset so it doesn't sink below plane

            glm::mat4 mBall(1.f);
            mBall = glm::translate(mBall, 
                      glm::vec3(ballLine.x, ballLine.y + yOffset, ballLine.z));

            // orientation (Rodrigues style or standard rotate)
            glm::vec3 axis    = glm::normalize(glm::vec3(0,1,0));
            float angleRad    = glm::radians(ballLine.oriY); 
            glm::mat4 rot     = glm::rotate(glm::mat4(1.f), angleRad, axis);
            mBall = mBall * rot;

            // scale
            mBall = glm::scale(mBall, glm::vec3(sx, sy, sz));

            glUniformMatrix4fv(uM,1,GL_FALSE, glm::value_ptr(mBall));
            glBindVertexArray(g_ballMesh.vao);
            glDrawElements(GL_TRIANGLES, g_ballMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        // --------------------------
        // draw goal
        // --------------------------
        glUniform1i(uObj, 2); // goal
        {
            glm::vec3 gSize = g_goalBox.vmax - g_goalBox.vmin;
            float ow= gSize.x;
            float oh= gSize.y;

            // *1.5f 
            float sx= (ow>1e-5f)? (g_goalData.width / ow) * 1.5f : 1.5f;
            float sy= (oh>1e-5f)? (g_goalData.height/oh) * 1.5f : 1.5f;
            float sz= sx;

            glm::mat4 mGoal(1.f);
            float footOffsetY= -(g_goalBox.vmin.y)* sy;
            mGoal= glm::translate(mGoal, 
                     glm::vec3(g_goalData.x, g_goalData.y + footOffsetY, g_goalData.z));

            // orientation
            float yaw= glm::radians(g_goalData.oriY);
            mGoal= glm::rotate(mGoal, glm::radians(180.0f), glm::vec3(0,1,0));

            mGoal= glm::scale(mGoal, glm::vec3(sx, sy, sz));

            glUniformMatrix4fv(uM,1,GL_FALSE, glm::value_ptr(mGoal));
            glBindVertexArray(g_goalMesh.vao);
            glDrawElements(GL_TRIANGLES, g_goalMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        // render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // swap
        glfwSwapBuffers(win);
    }

    // cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
