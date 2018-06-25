package edu.usc.thevillagers.serversideagent.command;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import edu.ucar.ral.nujan.hdf.HdfException;
import edu.ucar.ral.nujan.hdf.HdfFileWriter;
import edu.ucar.ral.nujan.hdf.HdfGroup;
import edu.usc.thevillagers.serversideagent.agent.Human;
import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import edu.usc.thevillagers.serversideagent.env.actuator.Actuator;
import edu.usc.thevillagers.serversideagent.env.actuator.Actuator.Reverser;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.command.WrongUsageException;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;
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
		try {
			File record = null;
			for(File file : new File("tmp/records/").listFiles()) {
				if(file.getName().contains(args[0])) {
					if(record != null) 
						throw new WrongUsageException(file.getName()+" and "+record.getName() + " match " + args[0]);
					record = file;
				}
			}
			final WorldRecordReplayer replayer = new WorldRecordReplayer(record);
			Class<?> envClass = envManager.findEnvClass(args[1]);
			new Thread(() -> {
				try {
					compile(replayer, (Environment) envClass.newInstance());
					server.addScheduledTask(() -> {
						sender.sendMessage(new TextComponentString("Compile success!"));
					});
				} catch (Exception e) {
					server.addScheduledTask(() -> {
						e.printStackTrace();
						sender.sendMessage(new TextComponentString("An error occured: "+e.toString()));
					});
				}
			}).start();
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
		List<List<Actuator.Reverser>> reversers = new ArrayList<>();
		for(Entity e : replay.world.loadedEntityList) {
			if(e instanceof EntityPlayerMP) {
				Human h = new Human(env, (EntityPlayerMP) e);
				humans.add(h);
				List<Actuator.Reverser> hReversers = new ArrayList<>();
				for(Actuator actuator : env.actuators)
					hReversers.add(actuator.reverser(h, replay));
				reversers.add(hReversers);
			}
		}
		int samples = (replay.duration + ((int) replay.worldTimeOffset % 5) + 4) / 5;
		int[] obsDim = {samples, humans.size(), env.observationDim};
		int[] actDim = {samples, humans.size(), env.actionDim};
		float[] obsBuffer = new float[obsDim[0] * obsDim[1] * obsDim[2]];
		float[] actBuffer = new float[actDim[0] * actDim[1] * actDim[2]]; //TODO use java.nio?
		long lastReport = System.currentTimeMillis();
		while(replay.currentTick < replay.duration) {
			int agentTick = (int) (replay.currentTick + (replay.worldTimeOffset % 5)) / 5;
			for(int h = 0; h < humans.size(); h ++) {
				Human human = humans.get(h);
				env.encodeObservation(human, human.observationVector);
				int offset = ((agentTick * obsDim[1]) + h) * obsDim[2];
				for(int i = 0; i < env.observationDim; i ++) {
					obsBuffer[offset + i] = human.observationVector[i];
				}
				for(Reverser reverser : reversers.get(h)) {
					reverser.startStep();
				}
			}
			do {
				replay.endReplayTick();
				for(int h = 0; h < humans.size(); h ++)
					for(Reverser reverser : reversers.get(h))
						reverser.tick();
			} while((replay.currentTick + replay.worldTimeOffset) % 5 != 0 && replay.currentTick < replay.duration);
			int offset = agentTick * actDim[1] * actDim[2];
			for(int h = 0; h < humans.size(); h ++) {
				for(Reverser reverser : reversers.get(h)) {
					for(float v : reverser.endStep())
						actBuffer[offset++] = v;
				}
			}
			if(System.currentTimeMillis() - lastReport > 1000 * 10)  {
				lastReport = System.currentTimeMillis();
				System.out.printf("compile: %.1f\n", (replay.currentTick * 100F / replay.duration));
			}
		}
		File file = new File("tmp/imitation/dataset.h5");
		file.getParentFile().mkdirs();
		HdfFileWriter writer = new HdfFileWriter(file.getAbsolutePath(), HdfFileWriter.OPT_ALLOW_OVERWRITE);
		
		HdfGroup obsVar = writer.getRootGroup().addVariable("obsVar", HdfGroup.DTYPE_FLOAT32, 0, obsDim, obsDim, 0F, 0);
		HdfGroup actVar = writer.getRootGroup().addVariable("actVar", HdfGroup.DTYPE_FLOAT32, 0, actDim, actDim, 0F, 0);
		writer.endDefine();
		obsVar.writeData(new int[] {0, 0, 0}, obsBuffer, true);
		actVar.writeData(new int[] {0, 0, 0}, actBuffer, true);
		writer.close();
		System.out.println("compile: done");
	}
}
