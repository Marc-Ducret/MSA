package edu.usc.thevillagers.serversideagent.command;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import edu.ucar.ral.nujan.hdf.HdfException;
import edu.usc.thevillagers.serversideagent.agent.Human;
import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.command.WrongUsageException;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;

/**
 * Command that compiles a recording into a observation-action dataset
 */
public class CommandCompile extends CommandBase {
	
	private final EnvironmentManager envManager;

	public CommandCompile(EnvironmentManager envManager) {
		MinecraftForge.EVENT_BUS.register(this);
		this.envManager = envManager;
	}
	
	@Override
	public String getName() {
		return "compile";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/compile <record> <env-type>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 2) throw new WrongUsageException(getUsage(sender));
		File record = null;
		for(File file : new File("tmp/records/").listFiles()) {
			if(file.getName().contains(args[0])) {
				if(record != null) 
					throw new WrongUsageException(file.getName()+" and "+record.getName() + " match " + args[0]);
				record = file;
			}
		}
		try {
			Class<?> envClass = envManager.findEnvClass(args[1]);
			compile(new WorldRecordReplayer(record), (Environment) envClass.newInstance());
		} catch (Exception e) {
			e.printStackTrace();
			throw new CommandException("An error occured: "+e.toString());
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	private void compile(WorldRecordReplayer replay, Environment env) throws HdfException, IOException, InterruptedException, ExecutionException {
		replay.readInfo();
		replay.seek(0);
		env.readPars(new float[]{});
		env.init((WorldServer) replay.world);
		List<Human> humans = new ArrayList<>();
		for(Entity e : replay.world.getLoadedEntityList()) {
			if(e instanceof EntityPlayerMP)
				humans.add(new Human(env, (EntityPlayerMP) e));
		}
		while(replay.currentTick < replay.duration) {
			for(Human h : humans) {
				env.encodeObservation(h, h.observationVector);
			}
			System.out.println("TICK: " + replay.currentTick);
			for(Human h : humans) {
				System.out.println(h.entity);
				for(float f : h.observationVector) System.out.print(f+" ");
				System.out.println();
			}
			replay.endReplayTick();
		}
//		File file = new File("tmp/imitation/dataset.h5");
//		file.getParentFile().mkdirs();
//		HdfFileWriter writer = new HdfFileWriter(file.getAbsolutePath(), HdfFileWriter.OPT_ALLOW_OVERWRITE);
//		int[] dim = {10};
//		HdfGroup var = writer.getRootGroup().addVariable("var", HdfGroup.DTYPE_FIXED32, 0, dim, dim, 0, 0);
//		writer.endDefine();
//		var.writeData(new int[] {0}, new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, true);
//		writer.close();
	}
}
