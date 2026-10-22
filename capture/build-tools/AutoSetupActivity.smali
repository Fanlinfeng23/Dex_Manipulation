.class public Lcom/tianji/xrobotoolkit/AutoSetupActivity;
.super Lcom/unity3d/player/UnityPlayerActivity;

# A per-process latch: never invert the three toggles twice. The PC waits for
# UIOperate.Awake and a nonce echoed by AndroidProxy before requesting defaults.
.field private static configured:Z

.method public constructor <init>()V
    .locals 0
    invoke-direct {p0}, Lcom/unity3d/player/UnityPlayerActivity;-><init>()V
    return-void
.end method

.method protected onNewIntent(Landroid/content/Intent;)V
    .locals 4
    invoke-super {p0, p1}, Lcom/unity3d/player/UnityPlayerActivity;->onNewIntent(Landroid/content/Intent;)V
    const-string v0, "tianji_probe"
    invoke-virtual {p1, v0}, Landroid/content/Intent;->getStringExtra(Ljava/lang/String;)Ljava/lang/String;
    move-result-object v2
    if-eqz v2, :defaults
    const-string v0, "AndroidProxy"
    const-string v1, "AndroidCall"
    invoke-static {v0, v1, v2}, Lcom/unity3d/player/UnityPlayer;->UnitySendMessage(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
    :defaults
    const-string v0, "tianji_defaults"
    const/4 v1, 0x0
    invoke-virtual {p1, v0, v1}, Landroid/content/Intent;->getBooleanExtra(Ljava/lang/String;Z)Z
    move-result v0
    if-eqz v0, :done
    sget-boolean v0, Lcom/tianji/xrobotoolkit/AutoSetupActivity;->configured:Z
    if-nez v0, :already
    const/4 v0, 0x1
    sput-boolean v0, Lcom/tianji/xrobotoolkit/AutoSetupActivity;->configured:Z
    const-string v1, "InternalToggle"
    const-string v2, ""
    const-string v0, "Head"
    invoke-static {v0, v1, v2}, Lcom/unity3d/player/UnityPlayer;->UnitySendMessage(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
    const-string v0, "Controller"
    invoke-static {v0, v1, v2}, Lcom/unity3d/player/UnityPlayer;->UnitySendMessage(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
    const-string v0, "transforToggle"
    invoke-static {v0, v1, v2}, Lcom/unity3d/player/UnityPlayer;->UnitySendMessage(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
    const-string v0, "UIOperate"
    const-string v1, "TcpConnect"
    const-string v2, "127.0.0.1"
    invoke-static {v0, v1, v2}, Lcom/unity3d/player/UnityPlayer;->UnitySendMessage(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
    const-string v0, "TianjiAutoSetup"
    const-string v1, "DEFAULTS_QUEUED_V1"
    invoke-static {v0, v1}, Landroid/util/Log;->i(Ljava/lang/String;Ljava/lang/String;)I
    goto :ack
    :already
    const-string v0, "TianjiAutoSetup"
    const-string v1, "DEFAULTS_ALREADY_APPLIED_V1"
    invoke-static {v0, v1}, Landroid/util/Log;->i(Ljava/lang/String;Ljava/lang/String;)I
    :ack
    const-string v0, "AndroidProxy"
    const-string v1, "AndroidCall"
    const-string v2, "TianjiAutoConfigured|v1"
    invoke-static {v0, v1, v2}, Lcom/unity3d/player/UnityPlayer;->UnitySendMessage(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
    :done
    return-void
.end method
