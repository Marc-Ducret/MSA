package edu.usc.thevillagers.serversideagent;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import edu.usc.thevillagers.serversideagent.request.RequestManager;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.event.FMLServerStartingEvent;
import net.minecraftforge.fml.common.event.FMLServerStoppingEvent;

@Mod(modid = ServerSideAgentMod.MODID, name = ServerSideAgentMod.NAME, version = ServerSideAgentMod.VERSION, serverSideOnly = true, acceptableRemoteVersions = "*")
public class ServerSideAgentMod {
    public static final String MODID = "serversideagent";
    public static final String NAME = "Server Side Agent";
    public static final String VERSION = "1.0";
    
    private EnvironmentManager envManager;
    private RequestManager reqManager;

    @EventHandler
    public void preInit(FMLPreInitializationEvent event) {
    }

    @EventHandler
    public void init(FMLInitializationEvent event) {
    	MinecraftForge.EVENT_BUS.register(this);
    }
    
    @EventHandler
    public void serverLoad(FMLServerStartingEvent event) throws IOException {
    	envManager = new EnvironmentManager();
    	reqManager = new RequestManager(envManager);
    	reqManager.startRequestServer(1337);
    	
    	event.registerServerCommand(new CommandEnvironment(envManager));
    	event.registerServerCommand(new CommandFastTick());
    	event.registerServerCommand(new CommandTPS());
    	event.registerServerCommand(new CommandRecord());
    }
    
    @EventHandler
    public void serverClosing(FMLServerStoppingEvent event) throws IOException {
    	envManager.clearEnvs();
    	reqManager.stopRequestServer();
    }
}
