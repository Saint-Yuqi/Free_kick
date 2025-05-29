// ---------------------------------------------------------------------------
// main.cpp — Soccer-ball free-flight & bounce demo (PBR grass + ImGui)
// ---------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES               // ensure M_PI on MSVC

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

// stb_image for texture loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ---------------------------------------------------------------------------
// 1.  GLSL shaders
// ---------------------------------------------------------------------------
static const char* VERT_SHADER = R"(
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNorm;
layout(location=2) in vec2 inUV;

uniform mat4 model, view, proj;

out vec3 vPos;
out vec3 vNorm;
out vec2 vUv;

void main()
{
    vec4 world = model * vec4(inPos,1.0);
    vPos  = world.xyz;
    vNorm = mat3(transpose(inverse(model))) * inNorm;
    vUv   = inUV;
    gl_Position = proj * view * world;
}
)";

static const char* FRAG_SHADER = R"(
#version 330 core
in  vec3 vPos;
in  vec3 vNorm;
in  vec2 vUv;
out vec4 FragColor;

uniform vec3  lightPos, camPos, lightColor;
uniform float roughness, metallic;
uniform bool  isGrass;
uniform float exposure;           // user slider

uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;

// --- Soccer colours --------------------------------------------------------
const vec3 WHITE = vec3(1.5);
const vec3 BLACK = vec3(0.05);

vec3 soccerColor(vec2 uv)
{
    vec2 C[12] = vec2[](
        vec2(0.50,0.50), vec2(0.25,0.70), vec2(0.75,0.70),
        vec2(0.25,0.30), vec2(0.75,0.30), vec2(0.10,0.50),
        vec2(0.90,0.50), vec2(0.50,0.10), vec2(0.50,0.90),
        vec2(0.15,0.15), vec2(0.85,0.85), vec2(0.15,0.85)
    );
    for(int i=0;i<12;i++)
        if(distance(uv,C[i])<0.15) return BLACK;
    return WHITE;
}

// --- GGX helpers -----------------------------------------------------------
float D_GGX(vec3 N,vec3 H,float a)
{
    float a2=a*a, NdotH=max(dot(N,H),0.0);
    float denom=(NdotH*NdotH*(a2-1.0)+1.0);
    return a2/(3.14159*denom*denom);
}
float G_Schlick(float NdotV,float k){return NdotV/(NdotV*(1.0-k)+k);}
float G_Smith(vec3 N,vec3 V,vec3 L,float a)
{
    float k=(a+1.0); k=k*k/8.0;
    return G_Schlick(max(dot(N,V),0.0),k)*G_Schlick(max(dot(N,L),0.0),k);
}
vec3  F_Schlick(float cosT,vec3 F0){return F0+(1.0-F0)*pow(1.0-cosT,5.0);}

void main()
{
    vec3 N = normalize(vNorm);
    vec3 V = normalize(camPos - vPos);
    vec3 L = normalize(lightPos - vPos);
    vec3 H = normalize(V + L);

    vec3 baseCol;
    float a = roughness;
    float m = metallic;

    if(isGrass)
    {
        vec2 uv = vUv * 50.0;
        baseCol = texture(albedoMap, uv).rgb;
        N       = normalize(texture(normalMap, uv).rgb * 2.0 - 1.0);
        a       = texture(roughnessMap, uv).r;
        m       = 0.0;
    }
    else
    {
        baseCol = soccerColor(vUv);
    }

    vec3 F0 = mix(vec3(0.04), baseCol, m);
    vec3  F = F_Schlick(max(dot(H,V),0.0), F0);
    float D = D_GGX(N,H,a);
    float G = G_Smith(N,V,L,a);

    vec3 spec = (D*G*F) /
                (4.0*max(dot(N,V),0.0)*max(dot(N,L),0.0)+0.001);
    vec3 diff = (1.0-F) * (1.0-m) * baseCol / 3.14159;

    vec3 colour = (diff + spec) * lightColor * max(dot(N,L),0.0);

    // simple ambient + exposure
    colour = exposure * (0.20 * baseCol + colour);

    FragColor = vec4(colour,1.0);
}
)";

// ---------------------------------------------------------------------------
// 2.  Geometry builders
// ---------------------------------------------------------------------------
void buildSphere(int sec,int st,std::vector<float>& vb,
                 std::vector<unsigned>& ib)
{
    for(int i=0;i<=st;++i){
        float phi=M_PI/2-i*(M_PI/st);
        float y=sin(phi), r=cos(phi);
        for(int j=0;j<=sec;++j){
            float th=j*(2*M_PI/sec);
            float x=r*cos(th), z=r*sin(th);
            vb.insert(vb.end(),
            { x,y,z,  x,y,z,
              (float)j/sec, (float)i/st });
        }
    }
    for(int i=0;i<st;++i)
      for(int j=0;j<sec;++j){
          unsigned a=i*(sec+1)+j, b=a+sec+1;
          ib.insert(ib.end(),{a,b,a+1,b,b+1,a+1});
      }
}
void buildPlane(std::vector<float>& vb,std::vector<unsigned>& ib,float s=100.f)
{
    float v[] = { -s,0,-s, 0,1,0, 0,0,
                   s,0,-s, 0,1,0, 1,0,
                   s,0, s, 0,1,0, 1,1,
                  -s,0, s, 0,1,0, 0,1 };
    unsigned idx[]={0,1,2,2,3,0};
    vb.insert(vb.end(),std::begin(v),std::end(v));
    ib.insert(ib.end(),std::begin(idx),std::end(idx));
}

// ---------------------------------------------------------------------------
// 3.  Tiny utility helpers
// ---------------------------------------------------------------------------
GLuint compile(GLenum type,const char* src)
{
    GLuint s=glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[512]; glGetShaderInfoLog(s,512,nullptr,log);
        std::cerr<<"shader error:\n"<<log<<'\n';}
    return s;
}
GLuint loadTex(const char* path)
{
    int w,h,c; stbi_set_flip_vertically_on_load(true);
    unsigned char* d=stbi_load(path,&w,&h,&c,0);
    if(!d){ std::cerr<<"texture fail: "<<path<<'\n'; return 0;}
    GLenum fmt=(c==1)?GL_RED:(c==3)?GL_RGB:GL_RGBA;
    GLuint id; glGenTextures(1,&id); glBindTexture(GL_TEXTURE_2D,id);
    glTexImage2D(GL_TEXTURE_2D,0,fmt,w,h,0,fmt,GL_UNSIGNED_BYTE,d);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    stbi_image_free(d); return id;
}

// ---------------------------------------------------------------------------
// 4.  Globals & physical parameters
// ---------------------------------------------------------------------------
float  g_gravity  = -9.81f;
glm::vec3 g_ballPos(0);
glm::vec3 g_v0(10.f,8.f,0.f);
float g_tAcc = 0.f;

bool  g_followCam = false;
float g_rough = 0.3f, g_metal = 0.0f;
float g_exposure = 1.5f;

// solid sphere size (must match scale)
const float BALL_SCALE  = 1.0f;
const float BALL_RADIUS = BALL_SCALE * 1.0f;

// bounce parameters
float g_restitution = 0.7f;
float g_frictionXZ  = 0.8f;

// ---------------------------------------------------------------------------
// 5.  Entry
// ---------------------------------------------------------------------------
int main()
{
    // ----- GLFW / GL context ----------------------------------------------
    if(!glfwInit()){ std::cerr<<"glfw init fail\n"; return -1;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    GLFWwindow* win=glfwCreateWindow(1280,720,"Soccer Demo",nullptr,nullptr);
    if(!win){glfwTerminate(); return -1;}
    glfwMakeContextCurrent(win);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr<<"glad fail\n"; return -1;}
    glfwSwapInterval(1);

    // ----- ImGui -----------------------------------------------------------
    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win,true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ----- Shaders ---------------------------------------------------------
    GLuint vs=compile(GL_VERTEX_SHADER,VERT_SHADER);
    GLuint fs=compile(GL_FRAGMENT_SHADER,FRAG_SHADER);
    GLuint prog=glCreateProgram();
    glAttachShader(prog,vs); glAttachShader(prog,fs); glLinkProgram(prog);

    // ----- Geometry (sphere + plane) --------------------------------------
    std::vector<float> vbS; std::vector<unsigned> ibS; buildSphere(64,64,vbS,ibS);
    GLuint vaoS,vboS,eboS; glGenVertexArrays(1,&vaoS);
    glGenBuffers(1,&vboS); glGenBuffers(1,&eboS);
    glBindVertexArray(vaoS);
    glBindBuffer(GL_ARRAY_BUFFER,vboS); glBufferData(GL_ARRAY_BUFFER,vbS.size()*4,vbS.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboS); glBufferData(GL_ELEMENT_ARRAY_BUFFER,ibS.size()*4,ibS.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*4,(void*)0);           glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*4,(void*)(3*4));       glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*4,(void*)(6*4));       glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    std::vector<float> vbP; std::vector<unsigned> ibP; buildPlane(vbP,ibP);
    GLuint vaoP,vboP,eboP; glGenVertexArrays(1,&vaoP);
    glGenBuffers(1,&vboP); glGenBuffers(1,&eboP);
    glBindVertexArray(vaoP);
    glBindBuffer(GL_ARRAY_BUFFER,vboP); glBufferData(GL_ARRAY_BUFFER,vbP.size()*4,vbP.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboP); glBufferData(GL_ELEMENT_ARRAY_BUFFER,ibP.size()*4,ibP.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*4,(void*)0);           glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*4,(void*)(3*4));       glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*4,(void*)(6*4));       glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    // ----- Textures --------------------------------------------------------
    GLuint texDiff=loadTex("grass_diffuse.png");
    GLuint texNorm=loadTex("grass_normal.png");
    GLuint texRgh =loadTex("grass_roughness.png");

    // ----- Uniforms --------------------------------------------------------
    glUseProgram(prog);
    GLint uM = glGetUniformLocation(prog,"model");
    GLint uV = glGetUniformLocation(prog,"view");
    GLint uP = glGetUniformLocation(prog,"proj");
    GLint uLP= glGetUniformLocation(prog,"lightPos");
    GLint uLC= glGetUniformLocation(prog,"lightColor");
    GLint uCP= glGetUniformLocation(prog,"camPos");
    GLint uR = glGetUniformLocation(prog,"roughness");
    GLint uMet=glGetUniformLocation(prog,"metallic");
    GLint uIs = glGetUniformLocation(prog,"isGrass");
    GLint uExp= glGetUniformLocation(prog,"exposure");

    glUniform3f(uLP,0.f,30.f,30.f);
    glUniform3f(uLC,3.f,3.f,3.f);

    glUniform1i(glGetUniformLocation(prog,"albedoMap"),0);
    glUniform1i(glGetUniformLocation(prog,"normalMap"),1);
    glUniform1i(glGetUniformLocation(prog,"roughnessMap"),2);

    int w,h; glfwGetFramebufferSize(win,&w,&h);
    glm::mat4 proj = glm::perspective(glm::radians(45.f),float(w)/h,0.1f,200.f);
    glUniformMatrix4fv(uP,1,GL_FALSE,glm::value_ptr(proj));

    glEnable(GL_DEPTH_TEST);

    // ----- Main loop -------------------------------------------------------
    double last = glfwGetTime(); float spin=0.f;

    while(!glfwWindowShouldClose(win))
    {
        // --- timing -------------------------------------------------------
        double now = glfwGetTime(); float dt=float(now-last); last=now;
        glfwPollEvents();

        // --- physics (Euler + bounce) -------------------------------------
        g_v0.y += g_gravity * dt;
        g_ballPos += g_v0 * dt;

        float bottom = g_ballPos.y - BALL_RADIUS;
        if(bottom < 0.0f){
            g_ballPos.y -= bottom;
            if(g_v0.y < 0.0f) g_v0.y = -g_v0.y * g_restitution;
            g_v0.x *= g_frictionXZ;
            g_v0.z *= g_frictionXZ;
            if(std::fabs(g_v0.y) < 0.05f) g_v0.y = 0.0f;
            if(glm::length(glm::vec2(g_v0)) < 0.05f) g_v0.x = g_v0.z = 0.0f;
        }

        spin += dt;

        // --- ImGui --------------------------------------------------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::Checkbox("Follow Camera",&g_followCam);
        if(ImGui::Button("Reset")){
            g_ballPos=glm::vec3(0); g_v0=glm::vec3(10,8,0);
        }
        ImGui::SliderFloat3("Velocity",&g_v0.x,-30.f,30.f);
        ImGui::SliderFloat("Restitution",&g_restitution,0.f,1.f);
        ImGui::SliderFloat("FrictionXZ",&g_frictionXZ ,0.f,1.f);
        ImGui::SliderFloat("Exposure",&g_exposure,0.1f,5.f);
        ImGui::SliderFloat("Roughness",&g_rough,0,1);
        ImGui::SliderFloat("Metallic",&g_metal,0,1);
        ImGui::Text("Ball (x,y,z) %.2f %.2f %.2f",
                    g_ballPos.x,g_ballPos.y,g_ballPos.z);
        ImGui::End();

        // --- camera -------------------------------------------------------
        glm::vec3 camPos;
        glm::mat4 view;
        if(g_followCam){
            camPos = g_ballPos + glm::vec3(0,2,8);
            view   = glm::lookAt(camPos,g_ballPos,glm::vec3(0,1,0));
        }else{
            camPos = glm::vec3(-10,5,20);
            view   = glm::lookAt(camPos,glm::vec3(0,1,0),glm::vec3(0,1,0));
        }

        // --- render pass --------------------------------------------------
        glClearColor(0.25f,0.3f,0.35f,1);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(uV,1,GL_FALSE,glm::value_ptr(view));
        glUniform3f(uCP,camPos.x,camPos.y,camPos.z);
        glUniform1f(uR,g_rough); glUniform1f(uMet,g_metal);
        glUniform1f(uExp,g_exposure);

        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,texDiff);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,texNorm);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D,texRgh);

        // plane
        glUniform1i(uIs,1);
        glm::mat4 mPlane(1);
        glUniformMatrix4fv(uM,1,GL_FALSE,glm::value_ptr(mPlane));
        glBindVertexArray(vaoP);
        glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0);

        // ball
        glUniform1i(uIs,0);
        glm::mat4 mBall(1);
        mBall = glm::translate(mBall,g_ballPos);
        mBall = glm::scale(mBall,glm::vec3(BALL_SCALE));
        mBall = glm::rotate(mBall,spin,glm::vec3(0,1,0));
        glUniformMatrix4fv(uM,1,GL_FALSE,glm::value_ptr(mBall));
        glBindVertexArray(vaoS);
        glDrawElements(GL_TRIANGLES,(GLsizei)ibS.size(),GL_UNSIGNED_INT,0);

        // gui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // ---- cleanup ---------------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
